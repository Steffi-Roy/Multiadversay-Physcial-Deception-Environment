"""Independent DDPG baseline.

Each agent runs standard single-agent DDPG with its own replay buffer.
The critic only sees the agent's own observation and action — no centralised
information.  Same network architecture as MADDPG for a fair comparison.
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Optional, Tuple

from config import Config
from networks import Actor, Critic
from buffer import ReplayBuffer
from maddpg import soft_update   # shared utility


class DDPGAgent:
    """Single independent DDPG agent."""

    def __init__(
        self,
        agent_idx: int,
        obs_dim: int,
        act_dim: int,
        cfg: Config,
    ):
        self.idx    = agent_idx
        self.cfg    = cfg
        self.device = torch.device(cfg.device)

        critic_in = obs_dim + act_dim

        self.actor         = Actor(obs_dim, act_dim, cfg.hidden_dim).to(self.device)
        self.target_actor  = Actor(obs_dim, act_dim, cfg.hidden_dim).to(self.device)
        self.critic        = Critic(critic_in, cfg.hidden_dim).to(self.device)
        self.target_critic = Critic(critic_in, cfg.hidden_dim).to(self.device)

        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.actor_opt  = torch.optim.Adam(self.actor.parameters(),  lr=cfg.lr_actor)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=cfg.lr_critic)

        self.buffer      = ReplayBuffer(cfg.buffer_size, obs_dim, act_dim)
        self.total_steps = 0


    def _noise_std(self) -> float:
        frac = min(1.0, self.total_steps / self.cfg.noise_decay_steps)
        return self.cfg.noise_std_start + frac * (
            self.cfg.noise_std_end - self.cfg.noise_std_start
        )

    # Action selection
    def get_action(self, obs: np.ndarray, explore: bool = True) -> np.ndarray:
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action = self.actor(obs_t).squeeze(0).cpu().numpy()
        if explore:
            noise  = np.random.normal(0, self._noise_std(), action.shape)
            action = np.clip(action + noise, 0.0, 1.0)
        return action.astype(np.float32)

    # Storage
    def store(
        self,
        obs: np.ndarray,
        act: np.ndarray,
        rew: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        self.buffer.add(obs, act, rew, next_obs, done)
        self.total_steps += 1

    # Update─

    def update(self) -> Optional[Tuple[float, float]]:
        if len(self.buffer) < self.cfg.batch_size:
            return None

        obs, acts, rews, next_obs, done = self.buffer.sample(self.cfg.batch_size)

        obs_t      = torch.FloatTensor(obs).to(self.device)
        acts_t     = torch.FloatTensor(acts).to(self.device)
        rews_t     = torch.FloatTensor(rews).to(self.device)
        next_obs_t = torch.FloatTensor(next_obs).to(self.device)
        done_t     = torch.FloatTensor(done).to(self.device)

        # Critic
        with torch.no_grad():
            next_acts_t = self.target_actor(next_obs_t)
            target_q    = rews_t + self.cfg.gamma * (1.0 - done_t) * self.target_critic(
                torch.cat([next_obs_t, next_acts_t], dim=1)
            )
        curr_q = self.critic(torch.cat([obs_t, acts_t], dim=1))
        c_loss = F.mse_loss(curr_q, target_q)

        self.critic_opt.zero_grad()
        c_loss.backward()
        self.critic_opt.step()

        # Actor
        a_loss = -self.critic(torch.cat([obs_t, self.actor(obs_t)], dim=1)).mean()

        self.actor_opt.zero_grad()
        a_loss.backward()
        self.actor_opt.step()

        # Soft target updates
        soft_update(self.target_actor,  self.actor,  self.cfg.tau)
        soft_update(self.target_critic, self.critic, self.cfg.tau)

        return c_loss.item(), a_loss.item()


# Multi-agent wrapper
class IndependentDDPG:
    """Runs independent DDPG for every agent — no centralised information."""

    def __init__(
        self,
        agent_names: List[str],
        obs_dims: List[int],
        act_dims: List[int],
        cfg: Config,
    ):
        self.agent_names = agent_names
        self.n_agents    = len(agent_names)
        self.agents      = [
            DDPGAgent(i, obs_dims[i], act_dims[i], cfg)
            for i in range(self.n_agents)
        ]

    # ── Same public interface as MADDPG ───────────────────────────────────────

    def get_actions(self, obs_dict: dict, explore: bool = True) -> dict:
        return {
            name: self.agents[i].get_action(obs_dict[name], explore)
            for i, name in enumerate(self.agent_names)
            if name in obs_dict
        }

    def store(
        self,
        obs_list: List[np.ndarray],
        act_list: List[np.ndarray],
        rew_list: List[float],
        next_obs_list: List[np.ndarray],
        done: bool,
    ) -> None:
        for i, agent in enumerate(self.agents):
            agent.store(obs_list[i], act_list[i], rew_list[i], next_obs_list[i], done)


    def update(self) -> List[Optional[Tuple[float, float]]]:
        return [agent.update() for agent in self.agents]
