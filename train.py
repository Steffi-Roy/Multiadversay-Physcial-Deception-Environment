"""Training script with CLI for MADDPG vs DDPG."""
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import argparse
import os
import numpy as np
import torch
from typing import Dict, List, Tuple, Union

from config import Config
from maddpg import MADDPG
from ddpg import IndependentDDPG
from env_wrapper import make_env, get_env_info, split_agents

Trainer = Union[MADDPG, IndependentDDPG]


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


# Single episode 

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


# Training loop 

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


# Save results 

def save_results(trainer: Trainer, history: Dict[str, list], algo: str, cfg: Config) -> None:
    os.makedirs("results", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    good_names, adv_names = split_agents(trainer.agent_names)

    # Save rewards arrays
    if good_names:
        good_rewards = np.array([history[n] for n in good_names])  # (n_good, n_episodes)
        np.save(f"results/{algo}_{cfg.run_tag}_rewards.npy", good_rewards)

    if adv_names:
        adv_rewards = np.array([history[n] for n in adv_names])    # (n_adv, n_episodes)
        np.save(f"results/{algo}_{cfg.run_tag}_adv_rewards.npy", adv_rewards)

    # Save actor checkpoints
    for i, agent in enumerate(trainer.agents):
        torch.save(
            agent.actor.state_dict(),
            f"checkpoints/{algo}_{cfg.run_tag}_agent{i}_actor.pt",
        )

    print(f"  Saved results and checkpoints for {algo} {cfg.run_tag}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Train MADDPG or DDPG agent")
    parser.add_argument("--algo",            type=str,   default="maddpg",
                        choices=["maddpg", "ddpg"],
                        help="Algorithm to train (default: maddpg)")
    parser.add_argument("--num-good",        type=int,   default=2,
                        help="Number of cooperative good agents (default: 2)")
    parser.add_argument("--num-adversaries", type=int,   default=1,
                        help="Number of adversary agents (default: 1)")
    parser.add_argument("--episodes",        type=int,   default=None,
                        help="Number of training episodes (default: from Config)")
    parser.add_argument("--seed",            type=int,   default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--device",          type=str,   default="cpu",
                        help="Torch device (default: cpu)")
    parser.add_argument("--force-retrain",   action="store_true",
                        help="Retrain even if results already exist")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg = Config(
        num_good       = args.num_good,
        num_adversaries= args.num_adversaries,
        seed           = args.seed,
        device         = args.device,
        num_episodes   = args.episodes if args.episodes is not None else Config.num_episodes,
    )

    algo = args.algo

    # Skip if results exist and not forced
    rewards_path = f"results/{algo}_{cfg.run_tag}_rewards.npy"
    if os.path.exists(rewards_path) and not args.force_retrain:
        print(f"Results already exist at {rewards_path}. Use --force-retrain to overwrite.")
        return

    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    env = make_env(cfg)
    agents, obs_dims, act_dims = get_env_info(env)
    env.close()

    good_names, adv_names = split_agents(agents)

    print("\n" + "=" * 60)
    print(f"  Training {algo.upper()} — {cfg.run_tag}")
    print("=" * 60)
    print(f"  Agents   : {agents}")
    print(f"  Obs dims : {dict(zip(agents, obs_dims))}")
    print(f"  Act dims : {dict(zip(agents, act_dims))}")
    print(f"  Episodes : {cfg.num_episodes}")
    print(f"  Device   : {cfg.device}")

    # Print MADDPG critic input dimensions
    total_obs = sum(obs_dims)
    total_act = sum(act_dims)
    total_critic_in = total_obs + total_act
    print(f"  MADDPG critic input: total_obs={total_obs} + total_act={total_act} = {total_critic_in}")
    print("=" * 60)

    if algo == "maddpg":
        trainer = MADDPG(agents, obs_dims, act_dims, cfg)
        label   = "MADDPG"
    else:
        trainer = IndependentDDPG(agents, obs_dims, act_dims, cfg)
        label   = "Independent DDPG"

    history = train(trainer, cfg, label)
    save_results(trainer, history, algo, cfg)


if __name__ == "__main__":
    main()
