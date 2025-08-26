"""
Log the log probability during the generation process.
"""

import os

import hydra
import numpy as np
import ray

os.environ["NCCL_DEBUG"] = "WARN"
os.environ["TOKENIZERS_PARALLELISM"] = "true"
# os.environ['TORCH_COMPILE_DISABLE'] = '1'

from pprint import pprint

import pandas as pd
from omegaconf import OmegaConf

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.utils import hf_tokenizer
from verl.utils.fs import copy_to_local
from verl.utils.hdfs_io import makedirs
from verl.utils.model import compute_position_id_with_mask
from verl.workers.fsdp_workers import ActorRolloutRefWorker
from verl.utils.device import is_cuda_available
from verl.trainer.ppo.core_algos import kl_penalty
from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role


@hydra.main(config_path="config", config_name="generation_dual", version_base=None)
def main(config):
    run_generation(config)

def run_generation(config) -> None:
    if not ray.is_initialized():
        # this is for local ray cluster
        ray.init(
            runtime_env={"env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN"}},
            num_cpus=config.ray_init.num_cpus,
        )

    ray.get(main_task.remote(config))

@ray.remote(num_cpus=1)
def main_task(config):
    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)

    # read dataset. Note that the dataset should directly contain chat template format (e.g., a list of dictionary)
    dataset = pd.read_parquet(config.data.path)
    chat_lst = dataset[config.data.prompt_key].tolist()

    chat_lst = [chat.tolist() for chat in chat_lst]

    if config.rollout.temperature == 0.0:
        assert config.data.n_samples == 1, "When temperature=0, n_samples must be 1."
    assert config.data.n_samples >= 1, "n_samples should always >= 1"

    total_samples = len(dataset)
    config_batch_size = config.data.batch_size
    num_batch = -(-total_samples // config_batch_size)
    output_lst = [[] for _ in range(config.data.n_samples)]
    wg_list = []

    assert len(config.model_paths) == 2, "Please provide exactly two model paths for comparison."

    local_path = copy_to_local(config.model_paths[0]) # models share the same tokenizer
    trust_remote_code = config.data.get("trust_remote_code", False)
    tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    global_pool_id = "global_pool"
    resource_pool_spec = {
        global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }
    mapping = {
        Role.ActorRollout: global_pool_id
    }
    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)
    resource_pool_manager.create_resource_pool()
    resource_pool_to_cls = {pool: {} for pool in resource_pool_manager.resource_pool_dict.values()}

    # create actor rollout workers
    for idx, model_path in enumerate(config.model_paths):
        config_i_dict = OmegaConf.to_container(config, resolve=True)

        if "model" not in config_i_dict:
            config_i_dict["model"] = {}

        config_i_dict["model"]["path"] = model_path
        config_i_dict["model"]["external_lib"] = None
        pprint(config_i_dict)
        config_i = OmegaConf.create(config_i_dict)

        resource_pool = resource_pool_manager.get_resource_pool(Role.ActorRollout)
        rollout_cls = RayClassWithInitArgs(cls=ray.remote(ActorRolloutRefWorker), config=config_i, role="actor_rollout")
        resource_pool_to_cls[resource_pool][f"rollout_model_{idx}"] = rollout_cls
    
    all_wg = {}
    wg_kwargs = {}

    for resource_pool, class_dict in resource_pool_to_cls.items():
        worker_dict_cls = create_colocated_worker_cls(class_dict)
        wg_dict = RayWorkerGroup(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls, device_name="cuda" if is_cuda_available else "npu", **wg_kwargs)
        spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
        all_wg.update(spawn_wg)

    rollout_wg_0 = all_wg["rollout_model_0"]
    rollout_wg_0.init_model()
    rollout_wg_1 = all_wg["rollout_model_1"]
    rollout_wg_1.init_model()

    wg_list = [rollout_wg_0, rollout_wg_1]

    for batch_idx in range(num_batch):
        print(f"[{batch_idx + 1}/{num_batch}] Start to process.")
        batch_chat_lst = chat_lst[batch_idx * config_batch_size : (batch_idx + 1) * config_batch_size]
        inputs = tokenizer.apply_chat_template(
            batch_chat_lst,
            add_generation_prompt=True,
            padding=True,
            truncation=True,
            max_length=config.rollout.prompt_length,
            return_tensors="pt",
            return_dict=True,
            tokenize=True,
        )
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        position_ids = compute_position_id_with_mask(attention_mask)
        batch_dict = {"input_ids": input_ids, "attention_mask": attention_mask, "position_ids": position_ids}

        data = DataProto.from_dict(batch_dict)
        data_padded, pad_size = pad_dataproto_to_divisor(data, wg_list[0].world_size)

        # START TO GENERATE FOR n_samples TIMES
        print(f"[{batch_idx + 1}/{num_batch}] Start to generate.")
        for n_sample in range(config.data.n_samples):
            
            outputs = []
            for wg in wg_list:
                output_padded = wg.generate_sequences(data_padded)
                output = unpad_dataproto(output_padded, pad_size=pad_size)
                outputs.append(output)
            
            # we padded logprobs with tokenizer.pad_token_id
            log_probs_model0_response0 = outputs[0].batch["rollout_log_probs"] # [batch_size, response_length]
            log_probs_model1_response1 = outputs[1].batch["rollout_log_probs"]

            valid_response_mask_0 = (log_probs_model0_response0 != tokenizer.pad_token_id).long()
            valid_response_mask_1 = (log_probs_model1_response1 != tokenizer.pad_token_id).long()

            output0_padded, pad_size_0 = pad_dataproto_to_divisor(outputs[0], wg_list[1].world_size)
            output1_padded, pad_size_1 = pad_dataproto_to_divisor(outputs[1], wg_list[0].world_size)

            # compute log probabilities for cross combinations
            log_probs_model0_response1_padded = wg_list[0].compute_log_prob(output1_padded)
            log_probs_model1_response0_padded = wg_list[1].compute_log_prob(output0_padded)
            log_probs_model0_response1 = unpad_dataproto(log_probs_model0_response1_padded, pad_size=pad_size_1)
            log_probs_model1_response0 = unpad_dataproto(log_probs_model1_response0_padded, pad_size=pad_size_0)

            log_probs_model0_response1 = log_probs_model0_response1.batch["old_log_probs"]
            log_probs_model1_response0 = log_probs_model1_response0.batch["old_log_probs"]

            log_prob_difference_0 = kl_penalty(log_probs_model0_response0, log_probs_model1_response0, "kl") # [batch_size, response_length]
            log_prob_difference_1 = kl_penalty(log_probs_model1_response1, log_probs_model0_response1, "kl")

            kl_0 = (log_prob_difference_0 * valid_response_mask_0).sum(dim=-1) / valid_response_mask_0.sum(dim=-1) # [batch_size]
            kl_1 = (log_prob_difference_1 * valid_response_mask_1).sum(dim=-1) / valid_response_mask_1.sum(dim=-1)

            kl_average = 0.5 * (kl_0 + kl_1)

            output_lst[n_sample].extend(kl_average.cpu().numpy().tolist())

    # convert output_lst from (n_samples, n_data) to (n_data, n_sampels)
    output_array = np.array(output_lst, dtype=object)
    output_array = np.transpose(output_lst, axes=(1, 0))

    # write to a new parquet
    output_dir = os.path.dirname(config.data.output_path)
    makedirs(output_dir, exist_ok=True)

    # save numpy array to .npy file
    np.save(config.data.output_path, output_array)


if __name__ == "__main__":
    main()
