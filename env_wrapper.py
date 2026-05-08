"""Environment creation and inspection utilities."""
import warnings
import numpy as np
from typing import List, Tuple

warnings.filterwarnings("ignore", category=DeprecationWarning)


def make_env(cfg, render_mode=None):
    """
    Always uses custom_simple_adversary which supports num_adversaries >= 1.
    The deception task (N landmarks, adversary must infer target) is preserved
    regardless of how many adversaries are used.
    """
    from custom_simple_adversary import parallel_env
    env = parallel_env(
        N=cfg.num_good,
        num_adversaries=cfg.num_adversaries,
        max_cycles=cfg.max_cycles,
        continuous_actions=True,
        render_mode=render_mode,
    )
    env.reset(seed=cfg.seed)
    return env


def get_env_info(env) -> Tuple[List[str], List[int], List[int]]:
    agents   = env.possible_agents
    obs_dims = [env.observation_space(a).shape[0] for a in agents]
    act_dims = [env.action_space(a).shape[0]      for a in agents]
    return agents, obs_dims, act_dims


def split_agents(agent_names: List[str]) -> Tuple[List[str], List[str]]:
    """Returns (good_names, adv_names)."""
    good = [n for n in agent_names if 'adversary' not in n]
    adv  = [n for n in agent_names if 'adversary'     in n]
    return good, adv
