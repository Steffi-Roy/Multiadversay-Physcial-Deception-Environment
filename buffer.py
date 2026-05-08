import numpy as np
from typing import List, Tuple


class MAReplayBuffer:
    """Centralised replay buffer for MADDPG.
    """

    def __init__(
        self,
        capacity: int,
        n_agents: int,
        obs_dims: List[int],
        act_dims: List[int],
    ):
        self.capacity = capacity
        self.n_agents = n_agents
        self.ptr = 0
        self.size = 0

        self.obs      = [np.zeros((capacity, d), dtype=np.float32) for d in obs_dims]
        self.acts     = [np.zeros((capacity, d), dtype=np.float32) for d in act_dims]
        self.rews     = [np.zeros((capacity, 1), dtype=np.float32) for _ in range(n_agents)]
        self.next_obs = [np.zeros((capacity, d), dtype=np.float32) for d in obs_dims]
        self.done     = np.zeros((capacity, 1), dtype=np.float32)

    def add(
        self,
        obs_list: List[np.ndarray],
        act_list: List[np.ndarray],
        rew_list: List[float],
        next_obs_list: List[np.ndarray],
        done: bool,
    ):
        i = self.ptr
        for k in range(self.n_agents):
            self.obs[k][i]      = obs_list[k]
            self.acts[k][i]     = act_list[k]
            self.rews[k][i]     = rew_list[k]
            self.next_obs[k][i] = next_obs_list[k]
        self.done[i] = float(done)
        self.ptr  = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return (
            [self.obs[k][idxs]      for k in range(self.n_agents)],
            [self.acts[k][idxs]     for k in range(self.n_agents)],
            [self.rews[k][idxs]     for k in range(self.n_agents)],
            [self.next_obs[k][idxs] for k in range(self.n_agents)],
            self.done[idxs],
        )

    def __len__(self) -> int:
        return self.size


class ReplayBuffer:
    """Standard single-agent replay buffer for independent DDPG."""

    def __init__(self, capacity: int, obs_dim: int, act_dim: int):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0

        self.obs      = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.acts     = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rews     = np.zeros((capacity, 1),       dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.done     = np.zeros((capacity, 1),       dtype=np.float32)

    def add(
        self,
        obs: np.ndarray,
        act: np.ndarray,
        rew: float,
        next_obs: np.ndarray,
        done: bool,
    ):
        i = self.ptr
        self.obs[i]      = obs
        self.acts[i]     = act
        self.rews[i]     = rew
        self.next_obs[i] = next_obs
        self.done[i]     = float(done)
        self.ptr  = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        idxs = np.random.randint(0, self.size, size=batch_size)
        return (
            self.obs[idxs],
            self.acts[idxs],
            self.rews[idxs],
            self.next_obs[idxs],
            self.done[idxs],
        )

    def __len__(self) -> int:
        return self.size
