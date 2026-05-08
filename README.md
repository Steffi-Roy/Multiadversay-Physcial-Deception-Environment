# MADDPG vs Independent DDPG — Physical Deception

Comparison of **MADDPG** (centralised training, decentralised execution) and **Independent DDPG** on a multi-agent physical deception task, using a custom extension of PettingZoo's `simple_adversary_v3` that supports configurable numbers of adversaries.

## Task

Good agents (blue) must cover the target landmark (green) to hide it from adversaries (red). Adversaries observe agent positions but not which landmark is the target, so they must infer it from good agent behaviour. Good agents are rewarded for covering the target and penalised if any adversary gets close.

## Demo

<video src="Demo.mp4" controls width="800">
  Your browser does not support the video tag.
</video>


https://github.com/user-attachments/assets/78f78ad5-13e7-4e3b-a7d4-dab4b3689408


## Setup

```bash
pip install -r requirements.txt
```

## Usage

**Train:**
```bash
python train.py --algo maddpg --num-good 2 --num-adversaries 1
python train.py --algo ddpg   --num-good 2 --num-adversaries 1
```

Add `--force-retrain` to overwrite existing checkpoints.

**Evaluate and record videos:**
```bash
python evaluate.py --algo maddpg --num-good 2 --num-adversaries 1 --record --adv-gains-reward
```

`--adv-gains-reward` filters for episodes where the adversary gets close to the target (useful for observing deception behaviour).

**Run all configs:**
```bash
python evaluate.py --all-configs
```

Prints a summary table and saves `results/summary.csv`.

## Configs tested

| Config | Env agents |
|---|---|
| `2g_1a` | 2 good, 1 adversary |
| `2g_2a` | 2 good, 2 adversaries |
| `3g_2a` | 3 good, 2 adversaries |


## Files

| File | Description |
|---|---|
| `custom_simple_adversary.py` | Extended MPE environment supporting `num_adversaries >= 1` |
| `maddpg.py` | MADDPG trainer — centralised critic sees all agents' obs + actions |
| `ddpg.py` | Independent DDPG — each agent has its own local critic |
| `networks.py` | Actor (outputs `[0,1]`) and Critic MLPs, 2 hidden layers × 64 units |
| `buffer.py` | Pre-allocated numpy replay buffers (joint for MADDPG, per-agent for DDPG) |
| `train.py` | Training script — saves checkpoints to `checkpoints/`, rewards to `results/` |
| `evaluate.py` | Evaluation, video recording, and summary metrics |
| `env_wrapper.py` | Environment factory |
| `config.py` | All hyperparameters in one dataclass |
| `main.py` | Standalone training + plotting (alternative to `train.py` + `evaluate.py`) |

## Reward structure

All rewards are negative distances — always ≤ 0 by construction.

- **Good agents:** `−min_dist(good → target) − min_dist(any_adversary → target)`
- **Adversary:** `−dist(self → target)`

## Key hyperparameters

| Parameter | Value |
|---|---|
| Episodes | 10,000 | //default
| Batch size | 1,024 |
| Buffer size | 1,000,000 |
| Actor/Critic LR | 1e-2 |
| Discount γ | 0.95 |
| Target update τ | 0.01 |
| Hidden dim | 64 |
| Exploration noise | 0.3 → 0.01 (linear decay over 100k steps) |
