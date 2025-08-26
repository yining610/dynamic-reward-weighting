EXPERIMENT_NAME="deepseek-7b-math500-acc-concise-format-333-lr1e-6-constant-baseline" LEARNING_RATE="1e-6" WEIGHTS="[0.334,0.333,0.333]" WARMUP_STYLE="constant" bash examples/deepseek_trainer/run_deepseek_multiobjective_baseline.sh

EXPERIMENT_NAME="deepseek-7b-math500-acc-concise-format-333-lr1e-6-constant-vanilla" LEARNING_RATE="1e-6" WEIGHTS="[0.334,0.333,0.333]" WARMUP_STYLE="constant" bash examples/deepseek_trainer/run_deepseek_multiobjective_vanilla.sh

EXPERIMENT_NAME="deepseek-7b-math500-acc-concise-format-333-lr1e-6-constant-optimization" LEARNING_RATE="1e-6" WEIGHTS="[0.334,0.333,0.333]" WARMUP_STYLE="constant" bash examples/deepseek_trainer/run_deepseek_multiobjective_optimization.sh
