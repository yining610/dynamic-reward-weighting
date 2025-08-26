EXPERIMENT_NAME="qwen3-8b-mathlighteval-acc-concise-format-333-lr1e-6-constant-baseline" LEARNING_RATE="1e-6" WEIGHTS="[0.334,0.333,0.333]" WARMUP_STYLE="constant" bash examples/mathlighteval_trainer/run_mathlighteval_multiobjective_baseline.sh

EXPERIMENT_NAME="qwen3-8b-mathlighteval-acc-concise-format-333-lr1e-6-constant-vanilla" LEARNING_RATE="1e-6" WEIGHTS="[0.334,0.333,0.333]" WARMUP_STYLE="constant" bash examples/mathlighteval_trainer/run_mathlighteval_multiobjective_vanilla.sh

EXPERIMENT_NAME="qwen3-8b-mathligheval-acc-concise-format-333-lr1e-6-constant-optimization" LEARNING_RATE="1e-6" WEIGHTS="[0.334,0.333,0.333]" WARMUP_STYLE="constant" bash examples/mathlighteval_trainer/run_mathlighteval_multiobjective_optimization.sh
