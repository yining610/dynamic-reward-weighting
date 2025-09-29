# Learning to Optimize Multi-Objective Alignment Through Dynamic Reward Weighting

<p align="center">
  <img src="https://img.shields.io/github/stars/yining610/dynamic-reward-weighting?style=social"/>
  <img src="https://img.shields.io/github/forks/yining610/dynamic-reward-weighting?style=social"/>
  <img src="https://img.shields.io/github/license/yining610/dynamic-reward-weighting?style=flat&v=2"/>
</p>

<p align="center">
  <b>Dynamic multi-objective reward weighting methods compatible with various online reinforcement learning algorithms, datasets, and model families</b><br>
  <a href="https://arxiv.org/abs/2509.11452"><b>Paper on arXiv</b></a>
</p>

---
## 📖 Folder Structure
```
steps/       // callable scripts for data preprocessing
verl/        // source code of models, algorithms, data structures, metrics, etc. 
examples/    // bash scripts to run jobs
data/        // pre-processed data used in experiments
```

## ⚙️ Environment 
We use the `Dockerfile` to build the environment. For more setup instructions, see the [verl environment setup guide](https://verl.readthedocs.io/en/latest/start/install.html)

We use Wandb to log experiments, so please log in before running them.

## 🚀 Experiment
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
We provide scripts to replicate the preliminary findings shown in Appendix A.2 in the paper
```
bash examples/preliminary_experiment/model_merge.sh
bash examples/preliminary_experiment/run_main_generation_dual.sh
```
Note that we need to merge the saved checkpoints from FSDP and Megatron backends to HuggingFace models first.

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

## 📚 Citation
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
