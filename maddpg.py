"""Multi-Agent Deep Deterministic Policy Gradient (MADDPG).
Reference: Lowe et al., "Multi-Agent Actor-Critic for Mixed Cooperative-Competitive
           Environments", NeurIPS 2017.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple

from config import Config
from networks import Actor, Critic
from buffer import MAReplayBuffer


# Utility 

def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    """Polyak averaging: θ_target ← τ·θ_source + (1−τ)·θ_target."""
    for t_p, s_p in zip(target.parameters(), source.parameters()):
        t_p.data.copy_(tau * s_p.data + (1.0 - tau) * t_p.data)


# Per agent networks + optimisers

class MADDPGAgent:
    """One agent's networks within MADDPG."""

    def __init__(
        self,
        agent_idx: int,
        obs_dim: int,
        act_dim: int,
        all_obs_dims: List[int],
        all_act_dims: List[int],
        cfg: Config,
    ):
        self.idx     = agent_idx
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.cfg     = cfg
        self.device  = torch.device(cfg.device)

        critic_in = sum(all_obs_dims) + sum(all_act_dims)

        self.actor         = Actor(obs_dim, act_dim, cfg.hidden_dim).to(self.device)
        self.target_actor  = Actor(obs_dim, act_dim, cfg.hidden_dim).to(self.device)
        self.critic        = Critic(critic_in, cfg.hidden_dim).to(self.device)
        self.target_critic = Critic(critic_in, cfg.hidden_dim).to(self.device)

        #parameters to targets at initialisation
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.actor_opt  = torch.optim.Adam(self.actor.parameters(),  lr=cfg.lr_actor)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=cfg.lr_critic)

    # Action selection

    def get_action(self, obs: np.ndarray, noise_std: float = 0.0) -> np.ndarray:
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action = self.actor(obs_t).squeeze(0).cpu().numpy()
        if noise_std > 0.0:
            action = np.clip(action + np.random.normal(0, noise_std, action.shape), 0.0, 1.0)
        return action.astype(np.float32)

    # Update (centralised training)
    def update(
        self,
        batch: Tuple,
        all_agents: List["MADDPGAgent"],
    ) -> Tuple[float, float]:
        obs_batch, act_batch, rew_batch, next_obs_batch, done_batch = batch

        obs_t      = [torch.FloatTensor(o).to(self.device) for o in obs_batch]
        act_t      = [torch.FloatTensor(a).to(self.device) for a in act_batch]
        rew_t      = torch.FloatTensor(rew_batch[self.idx]).to(self.device)
        next_obs_t = [torch.FloatTensor(o).to(self.device) for o in next_obs_batch]
        done_t     = torch.FloatTensor(done_batch).to(self.device)

        #Critic Loss
        with torch.no_grad():
            next_acts_t = [ag.target_actor(next_obs_t[ag.idx]) for ag in all_agents]
            next_in     = torch.cat(next_obs_t + next_acts_t, dim=1)
            target_q    = rew_t + self.cfg.gamma * (1.0 - done_t) * self.target_critic(next_in)

        curr_in  = torch.cat(obs_t + act_t, dim=1)
        curr_q   = self.critic(curr_in)
        c_loss   = F.mse_loss(curr_q, target_q)

        self.critic_opt.zero_grad()
        c_loss.backward()
        self.critic_opt.step()

        # Actor loss
        # Re-compute actions; only differentiate through *this* agent's actor.
        curr_acts = []
        for ag in all_agents:
            if ag.idx == self.idx:
                curr_acts.append(ag.actor(obs_t[ag.idx]))
            else:
                curr_acts.append(act_t[ag.idx].detach())

        actor_in = torch.cat(obs_t + curr_acts, dim=1)
        a_loss   = -self.critic(actor_in).mean()

        self.actor_opt.zero_grad()
        a_loss.backward()
        self.actor_opt.step()

        return c_loss.item(), a_loss.item()

    def soft_update_targets(self) -> None:
        soft_update(self.target_actor,  self.actor,  self.cfg.tau)
        soft_update(self.target_critic, self.critic, self.cfg.tau)


#Trainer (manages all agents + shared buffer)

class MADDPG:
    """MADDPG trainer — shared replay buffer, per-agent networks."""

    def __init__(
        self,
        agent_names: List[str],
        obs_dims: List[int],
        act_dims: List[int],
        cfg: Config,
    ):
        self.agent_names = agent_names
        self.n_agents    = len(agent_names)
        self.cfg         = cfg

        self.agents = [
            MADDPGAgent(i, obs_dims[i], act_dims[i], obs_dims, act_dims, cfg)
            for i in range(self.n_agents)
        ]
        self.buffer       = MAReplayBuffer(cfg.buffer_size, self.n_agents, obs_dims, act_dims)
        self.total_steps  = 0

    #Noise schedule

    def _noise_std(self) -> float:
        frac = min(1.0, self.total_steps / self.cfg.noise_decay_steps)
        return self.cfg.noise_std_start + frac * (
            self.cfg.noise_std_end - self.cfg.noise_std_start
        )

    #IndependentDDPG 
    def get_actions(self, obs_dict: dict, explore: bool = True) -> dict:
        noise = self._noise_std() if explore else 0.0
        return {
            name: self.agents[i].get_action(obs_dict[name], noise)
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
        self.buffer.add(obs_list, act_list, rew_list, next_obs_list, done)
        self.total_steps += 1

    def update(self) -> Optional[List[Tuple[float, float]]]:
        if len(self.buffer) < self.cfg.batch_size:
            return None
        batch  = self.buffer.sample(self.cfg.batch_size)
        losses = [ag.update(batch, self.agents) for ag in self.agents]
        for ag in self.agents:
            ag.soft_update_targets()
        return losses
