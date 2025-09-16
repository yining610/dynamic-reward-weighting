The is the repository documenting experiments for the paper: **Learning to Optimize Multi-objective Alignment Through Dynamic Reward Weighting**.

## Folder Structure
```
steps/       // callable scripts for data preprocessing
verl/        // source code of models, algorithms, data strcutures, metrics, etc. 
examples/    // bash scripts to run jobs
data/        // pre-processed data used in experiments
```

## Environment 
We use the `Dockerfile` to build the environment. For more setup instructions, see the [verl environment setup guide](https://verl.readthedocs.io/en/latest/start/install.html)

We use wandb to log experiments and please log in before running them.

## Experiment
We provide all the bash scripts used in our experiments in the `examples/` directory.

### Hypervolume-Guided Weight Adaptation
Take an example of training Qwen3 on GRPO:
```
bash examples/grpo_trainer/run_qwen3-8b_multiobjective_vanilla.sh
```

run experiments in batch:
```
bash examples/grpo_trainer/run_batch_vanilla.sh
```

### Gradient-Based Weight Optimization:
Take an example of training Qwen3 on GRPO: 
```
bash examples/grpo_trainer/run_qwen3-8b_multiobjective_optimization.sh
```

run experiments in batch:
```
bash examples/grpo_trainer/run_batch_optimization.sh
```

### Preliminary experiments:
We provide scripts to replicate the preliminary findings shown in Appendix A.2 of the paper
```
bash examples/preliminary_experiment/model_merge.sh
bash examples/preliminary_experiment/run_main_generation_dual.sh
```
Note that we need to merge the saved checkpoitns from FSDP and Megatron backends to huggingface models first.

### Result Analysis
We also provide analysis code in `analysis_fns.py` for analyzing and visualizing results, with examples in `analysis.ipynb`.

### Important Files
Some important files that we modified and added on Verl
```
verl/trainer/ppo/ray_trainer_vanilla.py
verl/trainer/ppo/ray_trainer_optimization.py
verl/trainer/main_generation_dual.py

verl/utils/reward_score/dynamic_math/*

verl/workers/reward_manager/multi_objective.py
verl/workers/reward_manager/multi_objective_optimization.py
verl/workers/fsdp_workers.py
```

## Citation
If you use our code, please cite the following paper:
```
@misc{lu2025learningoptimizemultiobjectivealignment,
      title={Learning to Optimize Multi-Objective Alignment Through Dynamic Reward Weighting}, 
      author={Yining Lu and Zilong Wang and Shiyang Li and Xin Liu and Changlong Yu and Qingyu Yin and Zhan Shi and Zixuan Zhang and Meng Jiang},
      year={2025},
      eprint={2509.11452},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2509.11452}, 
}
```
