set -x

WORKSPACE=$(dirname "$(dirname "$(dirname "$(realpath "${BASH_SOURCE[0]}")")")")
echo "Using workspace: $WORKSPACE"

PROJECT_NAME="verl_rloo_math_vanilla"

EXPERIMENT_NAME=${EXPERIMENT_NAME:-"qwen3-8b-math500-acc-concise-format-vanilla"}

REWARD_PATH="['$WORKSPACE/verl/utils/reward_score/dynamic_math/math_accuracy.py','$WORKSPACE/verl/utils/reward_score/dynamic_math/math_conciseness.py','$WORKSPACE/verl/utils/reward_score/dynamic_math/math_format.py']"
MAXIMIZE="[True,False,True]"
WEIGHTS=${WEIGHTS:-"[0.334,0.333,0.333]"}

TRAIN_FILES="$WORKSPACE/data/math500/train.parquet"
VALIDATE_FILES="$WORKSPACE/data/math500/val.parquet"
TEST_FILES="$WORKSPACE/data/math500/test.parquet"

INIT_MODEL="Qwen/Qwen3-8B"

GPUS_PER_NODE=8
NUM_NODES=1
MICRO_BATCH_SIZE_PER_GPU=8

EPOCH=90

# --------------------------

export NCCL_SOCKET_IFNAME=eth0
export VLLM_ATTENTION_BACKEND=XFORMERS
export RAY_DEDUP_LOGS=0

dexport CUDA_DEVICE_MAX_CONNECTIONS=1

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=rloo \
    train_strategy.method='vanilla' \
    +train_strategy.strategy_config.maximize=$MAXIMIZE \
    data.train_files=$TRAIN_FILES \
    data.val_files=$VALIDATE_FILES \
    +data.test_files=$TEST_FILES \
    data.train_batch_size=64 \
    data.max_prompt_length=2048 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=$INIT_MODEL \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE_PER_GPU \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE_PER_GPU \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE_PER_GPU \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    reward_model.reward_manager=multi_objective \
    custom_reward_function.path=$REWARD_PATH \
    +custom_reward_function.weights=$WEIGHTS \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=$GPUS_PER_NODE \
    trainer.nnodes=$NUM_NODES \
    trainer.save_freq=-1 \
    trainer.test_freq=5 \
    trainer.default_local_dir=$WORKSPACE/checkpoints/$PROJECT_NAME/$EXPERIMENT_NAME \
    trainer.total_epochs=$EPOCH $@