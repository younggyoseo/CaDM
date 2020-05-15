# Context-aware Dynamics Model for Generalization in Model-based Reinforcement Learning

This repository contains code for the paper
**"[Context-aware Dynamics Model for Generalization in Model-based Reinforcement Learning](https://sites.google.com/view/cadm)"** 

## Requirements

* `python3`
* `tensorflow==1.15.0`
* `gym==0.16.0`
* `baselines==0.1.5`
* `mujoco-py==2.0.2.9`

## Model-based RL

### Context-aware Dynamics Model

1. Vanilla DM + CaDM
```
python -m run_scripts.run_cadm_pets --dataset halfcheetah --policy_type CEM --n_candidate 200 \
--normalize_flag --ensemble_size 1 --n_particles 1 --deterministic_flag 1 --history_length 10 \
--future_length 10 --seed 0
```

2. PE-TS + CaDM
```
python -m run_scripts.run_cadm_pets --dataset halfcheetah --policy_type CEM --n_candidate 200 \
--normalize_flag --ensemble_size 5 --n_particles 20 --deterministic_flag 0 --history_length 10 \
--future_length 10 --seed 0
```

### Baselines

1. Vanilla DM
```
python -m run_scripts.run_pets --dataset halfcheetah --policy_type CEM --n_candidate 200 \
--normalize_flag --ensemble_size 1 --n_particles 1 --deterministic_flag 1 --seed 0
```

2. PE-TS
```
python -m run_scripts.run_pets --dataset halfcheetah --policy_type CEM --n_candidate 200 \
--normalize_flag --ensemble_size 5 --n_particles 20 --deterministic_flag 0 --seed 0
```

## Model-free RL

### Context-aware Dynamics Model

1. PPO + (Vanilla + CaDM)
```
python -m run_scripts.model_free.run_ppo_cadm --entropy_coeff 0.0 --lr 0.0005 \
--num_rollouts 10 --num_steps 200 --num_minibatches 4 --policy_type CEM --n_candidate 200 \
--normalize_flag --deterministic_flag 1 --ensemble_size 1 --n_particles 1 --history_length 10 \
--future_length 10 --load_path [saved_path] --seed 0
```

2. PPO + (PE-TS + CaDM)
```
python -m run_scripts.model_free.run_ppo_cadm --entropy_coeff 0.0 --lr 0.0005 \
--num_rollouts 10 --num_steps 200 --num_minibatches 4 --policy_type CEM --n_candidate 200 \
--normalize_flag --deterministic_flag 0 --ensemble_size 5 --n_particles 20 --history_length 10 \
--future_length 10 --load_path [saved_path] --seed 0
```

For example, `saved_path` looks like:
`data/HALFCHEETAH/NORMALIZED/CaDM/DET/CEM/CAND_200/H_10/F_10/BACK_COEFF_0.5/DIFF/BATCH_256/EPOCH_5/hidden_200_lr_0.001_horizon_30_seed_0/checkpoints/params_epoch_19`
