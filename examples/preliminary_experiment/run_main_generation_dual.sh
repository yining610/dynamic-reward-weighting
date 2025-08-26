set -x

WORKSPACE=$(dirname "$(dirname "$(dirname "$(realpath "${BASH_SOURCE[0]}")")")")
echo "Using workspace: $WORKSPACE"

DATA_PATH="$WORKSPACE/data/math500/test.parquet"

# --------------------------

export NCCL_SOCKET_IFNAME=eth0
export VLLM_ATTENTION_BACKEND=XFORMERS
export RAY_DEDUP_LOGS=0

dexport CUDA_DEVICE_MAX_CONNECTIONS=1

for model_name_0 in 001 010 100 5HALF 25HALF;
do
    for model_name_1 in 001 010 100 5HALF 25HALF;
    do
        if [ "$model_name_0" != "$model_name_1" ]; then
            for step in 20 40 60 80 100 120 140 160 180 200 220 240 260;
            do
                # Models for comparison has to be at the same step
                MODEL_PATH_0="$WORKSPACE/checkpoints/merged_models/qwen3_8b-math500-acc-concise-format-$model_name_0/global_step_${step}_actor"
                MODEL_PATH_1="$WORKSPACE/checkpoints/merged_models/qwen3_8b-math500-acc-concise-format-$model_name_1/global_step_${step}_actor"
                
                MODEL_PATH="['$MODEL_PATH_0','$MODEL_PATH_1']"
                OUTPUT_PATH="$WORKSPACE/results/npy/qwen3_8b-math500-acc-concise-format/${model_name_0}-${model_name_1}/global_step_${step}_actor.npy"

                echo "Comparing models: $model_name_0 and $model_name_1 at step $step"

                python3 -m verl.trainer.main_generation_dual \
                    data.path=$DATA_PATH \
                    data.output_path=$OUTPUT_PATH \
                    model_paths=$MODEL_PATH
            done
        else
            echo "Skipping same model pair: $model_name_0 and $model_name_1"
        fi
    done
done
