#!/bin/bash
EXPERIMENT_NAME="qwen3_8b-math500-acc-concise-format-333-poly" WEIGHTS="[0.334,0.333,0.333]" WARMUP_STYLE="polynomial" bash examples/grpo_trainer/run_qwen3-8b_multiobjective_baseline.sh

EXPERIMENT_NAME="qwen3_8b-math500-acc-concise-format-333-constant" WEIGHTS="[0.334,0.333,0.333]" WARMUP_STYLE="constant" bash examples/grpo_trainer/run_qwen3-8b_multiobjective_baseline.sh

EXPERIMENT_NAME="qwen3_8b-math500-acc-concise-format-25HALF-poly" WEIGHTS="[0.25,0.375,0.375]" WARMUP_STYLE="polynomial" bash examples/grpo_trainer/run_qwen3-8b_multiobjective_baseline.sh

EXPERIMENT_NAME="qwen3_8b-math500-acc-concise-format-25HALF-constant" WEIGHTS="[0.25,0.375,0.375]" WARMUP_STYLE="constant" bash examples/grpo_trainer/run_qwen3-8b_multiobjective_baseline.sh

EXPERIMENT_NAME="qwen3_8b-math500-acc-concise-format-5HALF-poly" WEIGHTS="[0.5,0.25,0.25]" WARMUP_STYLE="polynomial" bash examples/grpo_trainer/run_qwen3-8b_multiobjective_baseline.sh

EXPERIMENT_NAME="qwen3_8b-math500-acc-concise-format-5HALF-constant" WEIGHTS="[0.5,0.25,0.25]" WARMUP_STYLE="constant" bash examples/grpo_trainer/run_qwen3-8b_multiobjective_baseline.sh
