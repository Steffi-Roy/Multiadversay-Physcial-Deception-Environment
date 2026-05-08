import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Union

from config import Config
from maddpg import MADDPG
from ddpg import IndependentDDPG
from env_wrapper import make_env, get_env_info

Trainer = Union[MADDPG, IndependentDDPG]


# Warm-up

def warmup(trainer: Trainer, cfg: Config) -> None:
    env        = make_env(cfg)
    names      = trainer.agent_names
    step_count = 0

    print(f"  Warm-up: collecting {cfg.warmup_steps} random steps … ", end="", flush=True)

    while step_count < cfg.warmup_steps:
        obs_dict, _ = env.reset()

        for _ in range(cfg.max_cycles):
            actions = {name: env.action_space(name).sample() for name in env.agents}

            next_obs_dict, rew_dict, term_dict, trunc_dict, _ = env.step(actions)
            done = any(term_dict.values()) or any(trunc_dict.values())

            obs_list      = [obs_dict.get(n, np.zeros(env.observation_space(n).shape[0])) for n in names]
            act_list      = [actions.get(n, env.action_space(n).sample())                  for n in names]
            rew_list      = [rew_dict.get(n, 0.0)                                           for n in names]
            next_obs_list = [next_obs_dict.get(n, obs_list[i])                              for i, n in enumerate(names)]

            trainer.store(obs_list, act_list, rew_list, next_obs_list, done)
            step_count += 1
            obs_dict = next_obs_dict

            if not env.agents or done:
                break

    env.close()
    print("done.")


#  Single episode
def run_episode(
    env,
    trainer: Trainer,
    cfg: Config,
    explore: bool = True,
) -> Tuple[Dict[str, float], int]:
    obs_dict, _ = env.reset()
    names       = trainer.agent_names
    ep_rewards  = {n: 0.0 for n in names}

    for step in range(cfg.max_cycles):
        actions = trainer.get_actions(obs_dict, explore=explore)
        actions = {k: v.astype(np.float32) for k, v in actions.items()}

        next_obs_dict, rew_dict, term_dict, trunc_dict, _ = env.step(actions)
        done = any(term_dict.values()) or any(trunc_dict.values())

        obs_list      = [obs_dict.get(n, np.zeros(env.observation_space(n).shape[0])) for n in names]
        act_list      = [actions.get(n, env.action_space(n).sample())                  for n in names]
        rew_list      = [float(rew_dict.get(n, 0.0))                                   for n in names]
        next_obs_list = [next_obs_dict.get(n, obs_list[i])                              for i, n in enumerate(names)]

        trainer.store(obs_list, act_list, rew_list, next_obs_list, done)
        trainer.update()

        for i, n in enumerate(names):
            ep_rewards[n] += rew_list[i]

        obs_dict = next_obs_dict

        if not env.agents or done:
            break

    return ep_rewards, step + 1


#Training loop 

def train(trainer: Trainer, cfg: Config, label: str) -> Dict[str, list]:
    print(f"\n{'─'*60}")
    print(f"Training: {label}")
    print(f"  Agents  : {trainer.agent_names}")
    print(f"{'─'*60}")

    warmup(trainer, cfg)

    history: Dict[str, list] = {n: [] for n in trainer.agent_names}
    history["ep_lengths"] = []

    env = make_env(cfg)

    for ep in range(cfg.num_episodes):
        ep_rewards, ep_len = run_episode(env, trainer, cfg, explore=True)

        for n, r in ep_rewards.items():
            history[n].append(r)
        history["ep_lengths"].append(ep_len)

        if (ep + 1) % cfg.log_interval == 0:
            w          = min(cfg.log_interval, ep + 1)
            adv_names  = [n for n in trainer.agent_names if "adversary" in n]
            good_names = [n for n in trainer.agent_names if "adversary" not in n]
            avg_good   = np.mean([np.mean(history[n][-w:]) for n in good_names]) if good_names else 0.0
            avg_adv    = np.mean([np.mean(history[n][-w:]) for n in adv_names])  if adv_names  else 0.0
            avg_len    = np.mean(history["ep_lengths"][-w:])
            print(
                f"  Ep {ep+1:5d}/{cfg.num_episodes}"
                f"  |  good: {avg_good:8.3f}"
                f"  |  adv: {avg_adv:8.3f}"
                f"  |  len: {avg_len:.1f}"
            )

    env.close()
    print(f"  Training complete.\n")
    return history


#Evaluation 

def evaluate(trainer: Trainer, cfg: Config) -> Dict[str, float]:
    env         = make_env(cfg)
    all_rewards = {n: [] for n in trainer.agent_names}

    for _ in range(cfg.eval_episodes):
        ep_rewards, _ = run_episode(env, trainer, cfg, explore=False)
        for n, r in ep_rewards.items():
            all_rewards[n].append(r)

    env.close()
    return {n: float(np.mean(rs)) for n, rs in all_rewards.items()}




def _smooth(arr: list, window: int) -> np.ndarray:
    if len(arr) < window:
        return np.array(arr)
    return np.convolve(arr, np.ones(window) / window, mode="valid")


def plot_training_curves(
    maddpg_history: Dict[str, list],
    ddpg_history: Dict[str, list],
    agent_names: List[str],
    cfg: Config,
    suffix: str = "",
) -> None:
    adv_names  = [n for n in agent_names if "adversary" in n]
    good_names = [n for n in agent_names if "adversary" not in n]
    w          = cfg.rolling_window

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"MADDPG vs Independent DDPG — Physical Deception{suffix}", fontsize=13)

    palette = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for ax, group, title in zip(
        axes,
        [good_names, adv_names],
        ["Cooperative Agents", "Adversary"],
    ):
        ax.set_title(title)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Episode Reward (smoothed)")

        for j, name in enumerate(group):
            color = palette[j % len(palette)]
            sm_m  = _smooth(maddpg_history[name], w)
            sm_d  = _smooth(ddpg_history[name],   w)
            xs    = np.arange(w - 1, w - 1 + len(sm_m))
            ax.plot(xs, sm_m, color=color,         lw=1.8, label=f"MADDPG {name}")
            ax.plot(xs, sm_d, color=color, ls="--", lw=1.5, alpha=0.8, label=f"DDPG {name}")

        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    fname = f"training_curves{suffix.replace(' ', '_')}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    print(f"  Saved plot: {fname}")
    plt.show()


# Print Summary 
def print_summary(
    eval_maddpg: Dict[str, float],
    eval_ddpg: Dict[str, float],
    agent_names: List[str],
    history_maddpg: Dict[str, list],
    history_ddpg: Dict[str, list],
) -> None:
    adv_names  = [n for n in agent_names if "adversary" in n]
    good_names = [n for n in agent_names if "adversary" not in n]

    print("\n" + "=" * 60)
    print("  EVALUATION SUMMARY")
    print("=" * 60)

    for label, evals in [("MADDPG", eval_maddpg), ("DDPG (independent)", eval_ddpg)]:
        print(f"\n  {label}:")
        for n in agent_names:
            print(f"    {n:<20s}: {evals[n]:8.3f}")

    print("\n" + "─" * 60)
    print("  LAST-100-EPISODE TRAINING AVERAGE")
    print("─" * 60)
    for label, hist in [("MADDPG", history_maddpg), ("DDPG", history_ddpg)]:
        print(f"\n  {label}:")
        for n in agent_names:
            last = hist[n][-100:] if len(hist[n]) >= 100 else hist[n]
            print(f"    {n:<20s}: {np.mean(last):8.3f}")

    print("\n" + "─" * 60)
    print("  DECEPTION ANALYSIS  (heuristic: good_avg > adv_avg)")
    print("─" * 60)
    for label, evals in [("MADDPG", eval_maddpg), ("DDPG (independent)", eval_ddpg)]:
        avg_good = np.mean([evals[n] for n in good_names]) if good_names else 0.0
        avg_adv  = np.mean([evals[n] for n in adv_names])  if adv_names  else 0.0
        verdict  = "✓ successful deception" if avg_good > avg_adv else "✗ adversary competitive"
        print(f"  {label:<22s}: good={avg_good:.3f}  adv={avg_adv:.3f}  →  {verdict}")

    print("=" * 60)


#  Multi-adversary scaling----

def multi_adversary_comparison(cfg: Config) -> List[dict]:
    records = []

    for n_good in cfg.multi_n_good_list:
        print(f"\n{'='*60}")
        print(f"  Multi-agent comparison: N={n_good} cooperative agents")

        sub_cfg = Config(
            num_good=n_good,
            num_episodes=cfg.multi_num_episodes,
        )

        env_info = make_env(sub_cfg)
        agents, obs_dims, act_dims = get_env_info(env_info)
        env_info.close()

        maddpg      = MADDPG(agents, obs_dims, act_dims, sub_cfg)
        maddpg_hist = train(maddpg, sub_cfg, f"MADDPG N={n_good}")
        eval_maddpg = evaluate(maddpg, sub_cfg)

        ddpg        = IndependentDDPG(agents, obs_dims, act_dims, sub_cfg)
        ddpg_hist   = train(ddpg, sub_cfg, f"DDPG N={n_good}")
        eval_ddpg   = evaluate(ddpg, sub_cfg)

        good_names = [a for a in agents if "adversary" not in a]
        adv_names  = [a for a in agents if "adversary"     in a]

        records.append({
            "n_good":      n_good,
            "maddpg_good": np.mean([eval_maddpg[n] for n in good_names]),
            "maddpg_adv":  np.mean([eval_maddpg[n] for n in adv_names]),
            "ddpg_good":   np.mean([eval_ddpg[n]   for n in good_names]),
            "ddpg_adv":    np.mean([eval_ddpg[n]   for n in adv_names]),
        })

        plot_training_curves(maddpg_hist, ddpg_hist, agents, sub_cfg, suffix=f"_N{n_good}")

    ns = [r["n_good"] for r in records]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Scaling cooperative team: MADDPG vs Independent DDPG", fontsize=13)

    axes[0].set_title("Cooperative agents avg reward")
    axes[0].plot(ns, [r["maddpg_good"] for r in records], "o-",  label="MADDPG")
    axes[0].plot(ns, [r["ddpg_good"]   for r in records], "s--", label="DDPG")
    axes[0].set_xlabel("N cooperative agents")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].set_title("Adversary avg reward")
    axes[1].plot(ns, [r["maddpg_adv"] for r in records], "o-",  color="tab:red",    label="MADDPG")
    axes[1].plot(ns, [r["ddpg_adv"]   for r in records], "s--", color="tab:orange", label="DDPG")
    axes[1].set_xlabel("N cooperative agents")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("multi_agent_scaling.png", dpi=150, bbox_inches="tight")
    print("  Saved plot: multi_agent_scaling.png")
    plt.show()

    return records


# #TODO: add args
def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="MADDPG vs DDPG — Physical Deception")
    parser.add_argument("--episodes",     type=int,   default=Config.num_episodes,
                        help="Run multi-adversary scaling comparison after main training")
    args = parser.parse_args()

    return Config(
        num_episodes        = args.episodes,

    )


def main() -> None:
    cfg = parse_args()

    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    env = make_env(cfg)
    agents, obs_dims, act_dims = get_env_info(env)
    env.close()

    print("\n" + "=" * 60)
    print("  Physical Deception — MADDPG vs Independent DDPG")
    print("=" * 60)
    print(f"  Agents   : {agents}")
    print(f"  Obs dims : {dict(zip(agents, obs_dims))}")
    print(f"  Act dims : {dict(zip(agents, act_dims))}")
    print(f"  Episodes : {cfg.num_episodes}")
    print(f"  Device   : {cfg.device}")
    print("=" * 60)

    maddpg         = MADDPG(agents, obs_dims, act_dims, cfg)
    maddpg_history = train(maddpg, cfg, "MADDPG")
    eval_maddpg    = evaluate(maddpg, cfg)

    ddpg            = IndependentDDPG(agents, obs_dims, act_dims, cfg)
    ddpg_history    = train(ddpg, cfg, "Independent DDPG")
    eval_ddpg       = evaluate(ddpg, cfg)

    plot_training_curves(maddpg_history, ddpg_history, agents, cfg)
    print_summary(eval_maddpg, eval_ddpg, agents, maddpg_history, ddpg_history)

    if cfg.use_multi_adversary:
        records = multi_adversary_comparison(cfg)
        print("\n  Scaling results:")
        print(f"  {'N':>4}  {'MADDPG good':>12}  {'DDPG good':>10}  {'MADDPG adv':>11}  {'DDPG adv':>9}")
        for r in records:
            print(
                f"  {r['n_good']:>4}"
                f"  {r['maddpg_good']:>12.3f}"
                f"  {r['ddpg_good']:>10.3f}"
                f"  {r['maddpg_adv']:>11.3f}"
                f"  {r['ddpg_adv']:>9.3f}"
            )


if __name__ == "__main__":
    main()
