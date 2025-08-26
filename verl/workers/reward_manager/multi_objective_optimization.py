from collections import defaultdict

import torch

from verl import DataProto
from verl.workers.reward_manager import MultiObjectiveRewardManager

class MultiObjectiveOptimalRewardManager(MultiObjectiveRewardManager):
    """The multi-objective reward manager for gradient-based weight optimization."""

    def __init__(self, tokenizer, num_examine, compute_score: dict, reward_fn_key="data_source", **reward_kwargs) -> None:
        super().__init__(tokenizer, num_examine, compute_score, reward_fn_key, **reward_kwargs)

    def __call__(self, data: DataProto, return_dict=False):

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]
            
        if data.meta_info["validate"]:
            reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
            weights: list = data.meta_info["weights"]
        else:
            reward_tensor= {
                f"reward_tensor_{reward_fn_name}": torch.zeros_like(data.batch["responses"], dtype=torch.float32) for reward_fn_name in self.compute_score.keys()
                }
        # store the score for each reward function
        reward_extra_info = defaultdict(list)

        already_print_data_sources = {}

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch["prompts"]

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            data_source = data_item.non_tensor_batch[self.reward_fn_key]
            
            scores = []
            for reward_fn_name, reward_fn in self.compute_score.items():
                score = None

                if reward_fn_name == "accuracy":
                    score = reward_fn(
                        solution_str=response_str,
                        ground_truth=ground_truth
                    )
                
                if reward_fn_name == "concise":
                    if data.meta_info["validate"]:
                        score = -1.0 # placeholder for validation, as we don't have global response tokens in validation
                    else:
                        assert "global_avg_tokens" in data.meta_info['extra_info'], "average number of tokens not found in batch meta info."
                        score = reward_fn(
                            solution_str=response_str,
                            global_avg_tokens=data.meta_info['extra_info']['global_avg_tokens'],
                            tokenizer=self.tokenizer
                        )

                if reward_fn_name == "format":
                    score = reward_fn(
                        solution_str=response_str
                    )
                
                if not data.meta_info["validate"]:
                    reward_tensor[f"reward_tensor_{reward_fn_name}"][i, valid_response_length - 1] = score

                reward_extra_info[reward_fn_name].append(score)
                scores.append(score)

            if any(score is None for score in scores):
                raise NotImplementedError("Unsupported reward function found.")
            
            if data.meta_info["validate"]:
                reward = sum(w * s for w, s in zip(weights, scores))
                reward_tensor[i, valid_response_length - 1] = reward

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                for reward_fn_name, score in zip(self.compute_score.keys(), scores):
                    print(f"[{reward_fn_name}]", score)

        if return_dict:
            # only return reward_extra_info if multiple reward functions are used
            reward_extra_info = reward_extra_info if len(self.compute_score) > 1 else defaultdict(list)

            return {
                "reward_extra_info": reward_extra_info,
                "reward_tensor": reward_tensor
            }
        else:
            raise NotImplementedError("Not return dict is not supported in this reward manager.")
