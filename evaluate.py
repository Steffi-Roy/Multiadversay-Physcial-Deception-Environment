"""Evaluation, video recording, metrics, and plots for MADDPG vs DDPG."""
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import argparse
import csv
import itertools
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from config import Config
from maddpg import MADDPG
from ddpg import IndependentDDPG
from env_wrapper import make_env, get_env_info, split_agents
from typing import Union

Trainer = Union[MADDPG, IndependentDDPG]


#Model loading 

def load_trainer(algo: str, cfg: Config) -> Trainer:
    env = make_env(cfg)
    agents, obs_dims, act_dims = get_env_info(env)
    env.close()

    if algo == "maddpg":
        trainer = MADDPG(agents, obs_dims, act_dims, cfg)
    else:
        trainer = IndependentDDPG(agents, obs_dims, act_dims, cfg)

    for i, agent in enumerate(trainer.agents):
        ckpt_path = f"checkpoints/{algo}_{cfg.run_tag}_agent{i}_actor.pt"
        if os.path.exists(ckpt_path):
            agent.actor.load_state_dict(
                torch.load(ckpt_path, map_location=cfg.device)
            )
        else:
            print(f"  Warning: checkpoint not found: {ckpt_path}")

    return trainer


#  Behavioral metrics 

def adversary_reach_rate(positions_log: List, threshold: float = 0.1) -> float:
    """% of eval episodes where ANY adversary step had distance < threshold to target landmark[0]."""
    if not positions_log:
        return 0.0
    reached = 0
    for ep_steps in positions_log:
        ep_reached = False
        for adv_positions, target_pos in ep_steps:
            for adv_pos in adv_positions:
                dist = np.linalg.norm(np.array(adv_pos) - np.array(target_pos))
                if dist < threshold:
                    ep_reached = True
                    break
            if ep_reached:
                break
        if ep_reached:
            reached += 1
    return reached / len(positions_log)


def coverage_spread(good_positions: List[np.ndarray]) -> float:
    """Mean pairwise Euclidean distance between cooperative agents."""
    if len(good_positions) < 2:
        return 0.0
    dists = [
        np.linalg.norm(np.array(a) - np.array(b))
        for a, b in itertools.combinations(good_positions, 2)
    ]
    return float(np.mean(dists))


def convergence_step(
    good_rewards_per_ep: List[float],
    threshold: float = 0.5,
    window: int = 100,
) -> Optional[int]:
    """First episode where the rolling mean of good-agent reward (over `window` episodes)
    exceeds `threshold`.  Rewards are always <= 0 (negative distances), so threshold
    must be negative — default -1.5 means agents are within ~1.5 units of the target
    on average."""
    for i in range(window - 1, len(good_rewards_per_ep)):
        rolling_mean = np.mean(good_rewards_per_ep[i - window + 1 : i + 1])
        if rolling_mean > threshold:
            return i
    return None


# Evaluation loop

def run_eval(
    trainer,
    cfg: Config,
    render: bool = False,
) -> Tuple[Dict[str, List[float]], List, List[float]]:
    """Run cfg.eval_episodes episodes with explore=False.

    Returns:
        rewards_dict   - {agent_name: [ep_reward, ...]}
        positions_log  - per-episode list of step-level (adv_positions, target_pos) tuples
        spreads_list   - per-episode coverage spread of good agents at final step
    """
    render_mode = "human" if render else None
    env = make_env(cfg, render_mode=render_mode)
    names = trainer.agent_names

    rewards_dict: Dict[str, List[float]] = {n: [] for n in names}
    positions_log: List[List[Tuple]] = []
    spreads_list: List[float] = []

    for _ep in range(cfg.eval_episodes):
        obs_dict, _ = env.reset(seed=cfg.seed + _ep + 1)
        ep_rewards = {n: 0.0 for n in names}
        ep_pos_log: List[Tuple] = []
        final_good_pos: List[np.ndarray] = []

        for step in range(cfg.max_cycles):
            actions = trainer.get_actions(obs_dict, explore=False)
            actions = {k: v.astype(np.float32) for k, v in actions.items()}

            next_obs_dict, rew_dict, term_dict, trunc_dict, _ = env.step(actions)
            done = any(term_dict.values()) or any(trunc_dict.values())

            for n in names:
                ep_rewards[n] += float(rew_dict.get(n, 0.0))

            try:
                world = env.unwrapped.world
                target_pos     = world.landmarks[0].state.p_pos.copy()
                adv_positions  = [a.state.p_pos.copy() for a in world.agents if 'adversary' in a.name]
                good_positions = [a.state.p_pos.copy() for a in world.agents if 'adversary' not in a.name]
                ep_pos_log.append((adv_positions, target_pos))
                if done or step == cfg.max_cycles - 1:
                    final_good_pos = good_positions
            except Exception:
                pass

            obs_dict = next_obs_dict

            if not env.agents or done:
                break

        for n in names:
            rewards_dict[n].append(ep_rewards[n])

        positions_log.append(ep_pos_log)
        if final_good_pos:
            spreads_list.append(coverage_spread(final_good_pos))

    env.close()
    return rewards_dict, positions_log, spreads_list


# Video recording

def record_episodes(
    trainer,
    cfg: Config,
    algo: str,
    n_episodes: int = 10,
    fps: int = 60,
    seconds: int = 10,
    adv_gains_reward: bool = False,
    max_attempts: int = 200,
) -> None:
    try:
        import imageio
        import pygame
        from PIL import Image, ImageDraw
    except ImportError as e:
        print(f"  Cannot record videos — missing dependency: {e}")
        return

    os.makedirs("videos", exist_ok=True)
    names = trainer.agent_names
    good_names, adv_names = split_agents(names)

    record_cycles = seconds * fps          # frames = duration × fps
    record_cfg    = Config(**{**cfg.__dict__, "max_cycles": record_cycles})

    saved   = 0
    attempt = 0

    if adv_gains_reward:
        print(f"  Filtering for episodes where adversary gains reward (max {max_attempts} attempts) …")

    while saved < n_episodes and attempt < max_attempts:
        env = make_env(record_cfg, render_mode="human")
        obs_dict, _ = env.reset(seed=cfg.seed + attempt + 1)
        frames      = []
        coop_cumrew = 0.0
        adv_cumrew  = 0.0

        for step in range(record_cycles):
            actions = trainer.get_actions(obs_dict, explore=False)
            actions = {k: v.astype(np.float32) for k, v in actions.items()}

            next_obs_dict, rew_dict, term_dict, trunc_dict, _ = env.step(actions)
            done = any(term_dict.values()) or any(trunc_dict.values())

            for n in good_names:
                coop_cumrew += float(rew_dict.get(n, 0.0))
            for n in adv_names:
                adv_cumrew += float(rew_dict.get(n, 0.0))

            try:
                env.render()
                pygame.event.pump()
                surface = pygame.display.get_surface()
                if surface is not None:
                    raw   = pygame.surfarray.array3d(surface)
                    frame = raw.transpose([1, 0, 2])
                    img   = Image.fromarray(frame)
                    draw  = ImageDraw.Draw(img)
                    draw.text((6, 6),  f"Ep {saved+1}  step {step}", fill=(255, 255, 255))
                    draw.text((6, 20), f"coop: {coop_cumrew:+.1f}",   fill=(255, 255, 255))
                    draw.text((6, 34), f"adv:  {adv_cumrew:+.1f}",    fill=(255, 255, 255))
                    frames.append(np.array(img))
            except Exception:
                pass

            obs_dict = next_obs_dict
            if not env.agents or done:
                break

        env.close()
        attempt += 1

        # Filter: skip episodes where adversary did not gain reward
        if adv_gains_reward and adv_cumrew <= 0:
            continue

        if frames:
            saved += 1
            out_path = f"videos/{algo}_{cfg.run_tag}_adv_ep{saved}.mp4"
            imageio.mimsave(out_path, frames, fps=fps)
            print(f"  [{saved}/{n_episodes}] saved {out_path}  (adv reward: {adv_cumrew:+.1f})")

    if saved < n_episodes:
        print(f"  Warning: only found {saved}/{n_episodes} matching episodes in {max_attempts} attempts.")


# Summary table 

def build_summary(base_cfg: Config) -> List[dict]:
    configs = [(2, 1), (2, 2), (3, 1), (3, 2)]
    algos   = ["maddpg", "ddpg"]
    rows    = []

    for (num_good, num_adv) in configs:
        for algo in algos:
            cfg      = Config(
                num_good       = num_good,
                num_adversaries= num_adv,
                seed           = base_cfg.seed,
                device         = base_cfg.device,
                eval_episodes  = base_cfg.eval_episodes,
            )
            run_tag  = cfg.run_tag
            rew_path = f"results/{algo}_{run_tag}_rewards.npy"
            adv_path = f"results/{algo}_{run_tag}_adv_rewards.npy"

            if not os.path.exists(rew_path):
                continue

            good_rewards = np.load(rew_path)        # (n_good, n_episodes)
            mean_per_ep  = good_rewards.mean(axis=0) # (n_episodes,)
            coop_last100 = float(mean_per_ep[-100:].mean())

            adv_last100 = None
            if os.path.exists(adv_path):
                adv_rewards  = np.load(adv_path)                       # (n_adv, n_episodes)
                adv_last100  = float(adv_rewards.mean(axis=0)[-100:].mean())

            conv_step = convergence_step(mean_per_ep.tolist())

            adv_reach_rate = None
            spread         = None

            # Check if checkpoints exist for eval
            has_ckpts = all(
                os.path.exists(f"checkpoints/{algo}_{run_tag}_agent{i}_actor.pt")
                for i in range(num_good + num_adv)
            )

            if has_ckpts:
                try:
                    trainer = load_trainer(algo, cfg)
                    rewards_dict, positions_log, spreads_list = run_eval(trainer, cfg)
                    good_names, adv_names = split_agents(trainer.agent_names)

                    adv_reach_rate = adversary_reach_rate(positions_log)
                    spread         = float(np.mean(spreads_list)) if spreads_list else None
                except Exception as exc:
                    print(f"  Warning: eval failed for {algo} {run_tag}: {exc}")

            rows.append({
                "config":       f"{num_good}g_{num_adv}a",
                "algo":         algo.upper(),
                "coop_last100": coop_last100,
                "adv_last100":  adv_last100,
                "adv_reach":    adv_reach_rate,
                "spread":       spread,
                "conv_step":    conv_step,
            })

    return rows


def print_and_save_summary(rows: List[dict]) -> None:
    os.makedirs("results", exist_ok=True)

    header = f"{'Config':<12}  {'Algo':<8}  {'Coop Reward (last 100)':>22}  {'Adv Reward (last 100)':>21}  {'Adv Reach Rate':>14}  {'Coverage Spread':>15}  {'Convergence Step':>16}"
    sep    = "─" * len(header)
    print("\n" + sep)
    print(header)
    print(sep)

    for r in rows:
        adv_rew   = f"{r['adv_last100']:.3f}" if r['adv_last100'] is not None else "  N/A"
        adv_reach = f"{r['adv_reach']:.3f}"   if r['adv_reach']   is not None else "  N/A"
        spread    = f"{r['spread']:.3f}"       if r['spread']      is not None else "  N/A"
        conv      = str(r['conv_step'])        if r['conv_step']   is not None else "  N/A"
        print(
            f"{r['config']:<12}  {r['algo']:<8}  {r['coop_last100']:>22.3f}"
            f"  {adv_rew:>21}  {adv_reach:>14}  {spread:>15}  {conv:>16}"
        )
    print(sep)

    csv_path = "results/summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["config", "algo", "coop_last100", "adv_last100", "adv_reach", "spread", "conv_step"],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved summary: {csv_path}")


# Combined learning curves 

def plot_all_learning_curves(base_cfg: Config) -> None:
    import matplotlib.pyplot as plt

    configs = [(2, 1), (2, 2), (3, 1), (3, 2)]
    algos   = ["maddpg", "ddpg"]

    color_map = {
        (2, 1): ("royalblue",   "cornflowerblue"),
        (2, 2): ("darkorange",  "sandybrown"),
        (3, 1): ("purple",      "plum"),
        (3, 2): ("forestgreen", "mediumseagreen"),
    }
    w = base_cfg.rolling_window

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("All Configs — Learning Curves: MADDPG vs DDPG", fontsize=13)

    for ax, side, title in zip(axes, ["good", "adv"], ["Cooperative Reward", "Adversary Reward"]):
        ax.set_title(title)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Episode Reward (smoothed)")

        for (num_good, num_adv), (maddpg_color, ddpg_color) in color_map.items():
            cfg     = Config(num_good=num_good, num_adversaries=num_adv)
            run_tag = cfg.run_tag
            suffix  = f"{num_good}g{num_adv}a"

            for algo, color, ls in [("maddpg", maddpg_color, "-"), ("ddpg", ddpg_color, "--")]:
                if side == "good":
                    path = f"results/{algo}_{run_tag}_rewards.npy"
                else:
                    path = f"results/{algo}_{run_tag}_adv_rewards.npy"

                if not os.path.exists(path):
                    continue

                arr      = np.load(path)                    # (n_agents, n_episodes)
                mean_ep  = arr.mean(axis=0)                 # (n_episodes,)
                smoothed = np.convolve(mean_ep, np.ones(w) / w, mode="valid")
                xs       = np.arange(w - 1, w - 1 + len(smoothed))

                ax.plot(
                    xs, smoothed,
                    color=color,
                    linestyle=ls,
                    lw=1.8,
                    label=f"{algo.upper()} {suffix}",
                )

        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    out_path = "results/all_configs_learning_curves.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  Saved plot: {out_path}")
    plt.close()


#Cli─

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate MADDPG or DDPG agents")
    parser.add_argument("--algo",            type=str,  default="maddpg",
                        choices=["maddpg", "ddpg"],
                        help="Algorithm to evaluate (default: maddpg)")
    parser.add_argument("--num-good",        type=int,  default=2,
                        help="Number of cooperative good agents (default: 2)")
    parser.add_argument("--num-adversaries", type=int,  default=1,
                        help="Number of adversary agents (default: 1)")
    parser.add_argument("--seed",            type=int,  default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--device",          type=str,  default="cpu",
                        help="Torch device (default: cpu)")
    parser.add_argument("--record",             action="store_true",
                        help="Record mp4 videos of eval episodes")
    parser.add_argument("--record-episodes",   type=int, default=10,
                        help="Number of episodes to record (default: 10)")
    parser.add_argument("--adv-gains-reward",  action="store_true",
                        help="Only save episodes where the adversary gains positive reward")
    parser.add_argument("--render",          action="store_true",
                        help="Live pygame window during eval (no recording)")
    parser.add_argument("--all-configs",     action="store_true",
                        help="Build full summary table + combined learning curves for all 6 configs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    base_cfg = Config(
        num_good       = args.num_good,
        num_adversaries= args.num_adversaries,
        seed           = args.seed,
        device         = args.device,
    )

    if args.all_configs:
        print("\nBuilding summary for all configs …")
        rows = build_summary(base_cfg)
        print_and_save_summary(rows)
        plot_all_learning_curves(base_cfg)
        return

    algo = args.algo
    cfg  = base_cfg

    print(f"\nEvaluating {algo.upper()} — {cfg.run_tag}")
    trainer = load_trainer(algo, cfg)

    if args.record:
        record_episodes(
            trainer, cfg, algo,
            n_episodes        = args.record_episodes,
            fps               = 60,
            seconds           = 6,
            adv_gains_reward  = args.adv_gains_reward,
        )
    else:
        rewards_dict, positions_log, spreads_list = run_eval(trainer, cfg, render=args.render)

        good_names, adv_names = split_agents(trainer.agent_names)

        print(f"\n  Results over {cfg.eval_episodes} episodes:")
        for n in trainer.agent_names:
            mean_r = float(np.mean(rewards_dict[n]))
            print(f"    {n:<20s}: {mean_r:8.3f}")

        if positions_log:
            reach = adversary_reach_rate(positions_log)
            print(f"\n  Adversary reach rate : {reach:.3f}")

        if spreads_list:
            spread = float(np.mean(spreads_list))
            print(f"  Coverage spread      : {spread:.3f}")

        if good_names:
            mean_good_ep = [
                float(np.mean([rewards_dict[n][ep] for n in good_names]))
                for ep in range(cfg.eval_episodes)
            ]
            conv = convergence_step(mean_good_ep)
            print(f"  Convergence step     : {conv}")

        # Plot training curves from saved rewards
        import matplotlib.pyplot as plt
        w = cfg.rolling_window
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        fig.suptitle(f"{algo.upper()} — {cfg.run_tag} Training Curves", fontsize=13)

        for ax, side, title in zip(axes, ["good", "adv"], ["Cooperative Reward", "Adversary Reward"]):
            ax.set_title(title)
            ax.set_xlabel("Episode")
            ax.set_ylabel("Episode Reward (smoothed)")
            suffix = "_adv" if side == "adv" else ""
            path = f"results/{algo}_{cfg.run_tag}{suffix}_rewards.npy"
            if os.path.exists(path):
                arr      = np.load(path)
                mean_ep  = arr.mean(axis=0)
                smoothed = np.convolve(mean_ep, np.ones(w) / w, mode="valid")
                xs       = np.arange(w - 1, w - 1 + len(smoothed))
                ax.plot(xs, smoothed, lw=1.8)
            else:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.grid(alpha=0.3)

        plt.tight_layout()
        fname = f"results/{algo}_{cfg.run_tag}_curve.png"
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        print(f"  Saved plot: {fname}")
        plt.show()


if __name__ == "__main__":
    main()
