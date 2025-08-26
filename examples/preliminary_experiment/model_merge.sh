WORKSPACE=$(dirname "$(dirname "$(realpath "${BASH_SOURCE[0]}")")")
echo "Using workspace: $WORKSPACE"
export PYTHONPATH=$PYTHONPATH:"$WORKSPACE"

for name in 001 010 100 5HALF 25HALF;
do
    MODEL_NAME="qwen3_8b-math500-acc-concise-format-$name"
    MODEL_DIR="checkpoints/$MODEL_NAME"

    for step in 20 40 60 80 100 120 140 160 180 200 220 240 260;
    do
        if [ -d "$MODEL_DIR/global_step_$step/actor" ]; then

            CHECKPOITN_DIR="$MODEL_DIR/global_step_$step/actor"
            echo "Merging models for checkpoint $CHECKPOITN_DIR"

            python steps/model_merge.py merge \
                --backend fsdp \
                --local_dir $CHECKPOITN_DIR \
                --target_dir "checkpoints/merged_models/$MODEL_NAME/global_step_${step}_actor"

            rm -r "$MODEL_DIR/global_step_$step"
        else
            echo "Directory $MODEL_DIR/global_step_$step does not exist, skipping."
        fi
    done
done