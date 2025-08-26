set -x

WORKSPACE=$(dirname "$(dirname "$(dirname "$(realpath "${BASH_SOURCE[0]}")")")")
echo "Using workspace: $WORKSPACE"

PROJECT_NAME="verl_grpo_math_deepseek"

EXPERIMENT_NAME=${EXPERIMENT_NAME:-"deepseek-7b-math500-acc-concise-format-optimization"}

REWARD_PATH="['$WORKSPACE/verl/utils/reward_score/dynamic_math/math_accuracy.py','$WORKSPACE/verl/utils/reward_score/dynamic_math/math_conciseness.py','$WORKSPACE/verl/utils/reward_score/dynamic_math/math_format.py']"
WEIGHTS=${WEIGHTS:-"[0.334,0.333,0.333]"}

TRAIN_FILES="$WORKSPACE/data/math500/train.parquet"
TEST_FILES="$WORKSPACE/data/math500/test.parquet"

INIT_MODEL="deepseek-ai/deepseek-llm-7b-chat"

GPUS_PER_NODE=8
NUM_NODES=1
MICRO_BATCH_SIZE_PER_GPU=8

WEIGHT_ACCUMULATION_STEPS=1
MAX_GRAD_NORM=1.0
REGULARIZATION_FACTOR=1e-4
MIN_WEIGHT=0.0
MAX_WEIGHT=5.0
LEARNING_RATE=${LEARNING_RATE:-"1e-6"}
WARMUP_STYLE=${WARMUP_STYLE:-"polynomial"}

CLIP_RATIO_C=100 # set high to disable clipping, default 3
CLIP_RATIO_LOW=100 # set high to disable clipping, default 0.2
CLIP_RATIO_HIGH=100 # set high to disable clipping, default 0.2

EPOCH=90

# --------------------------

export NCCL_SOCKET_IFNAME=eth0
export VLLM_ATTENTION_BACKEND=XFORMERS
export RAY_DEDUP_LOGS=0

dexport CUDA_DEVICE_MAX_CONNECTIONS=1

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    train_strategy.method='optimization' \
    data.train_files=$TRAIN_FILES \
    data.val_files=$TEST_FILES \
    data.train_batch_size=64 \
    data.max_prompt_length=1024 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=$INIT_MODEL \
    actor_rollout_ref.actor.optim.lr=$LEARNING_RATE \
    actor_rollout_ref.actor.optim.warmup_style=$WARMUP_STYLE \
    actor_rollout_ref.actor.grad_clip=$MAX_GRAD_NORM \
    actor_rollout_ref.actor.clip_ratio_c=$CLIP_RATIO_C \
    actor_rollout_ref.actor.clip_ratio_low=$CLIP_RATIO_LOW \
    actor_rollout_ref.actor.clip_ratio_high=$CLIP_RATIO_HIGH \
    +actor_rollout_ref.actor.weight_accumulation_steps=$WEIGHT_ACCUMULATION_STEPS \
    +actor_rollout_ref.actor.regularization_factor=$REGULARIZATION_FACTOR \
    +actor_rollout_ref.actor.min_weight=$MIN_WEIGHT \
    +actor_rollout_ref.actor.max_weight=$MAX_WEIGHT \
    +actor_rollout_ref.actor.target_layer_ids=[10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25] \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE_PER_GPU \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
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
    reward_model.reward_manager=multi_objective_optimization \
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