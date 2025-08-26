#!/bin/bash
EXPERIMENT_NAME="qwen3-8b-math500-acc-concise-format-25HALF-vanilla" WEIGHTS="[0.25,0.375,0.375]" bash examples/grpo_trainer/run_qwen3-8b_multiobjective_vanilla.sh

EXPERIMENT_NAME="qwen3-8b-math500-acc-concise-format-5HALF-vanilla" WEIGHTS="[0.5,0.25,0.25]" bash examples/grpo_trainer/run_qwen3-8b_multiobjective_vanilla.sh

EXPERIMENT_NAME="qwen3-8b-math500-acc-concise-format-333-vanilla" WEIGHTS="[0.334,0.333,0.333]" bash examples/grpo_trainer/run_qwen3-8b_multiobjective_vanilla.sh