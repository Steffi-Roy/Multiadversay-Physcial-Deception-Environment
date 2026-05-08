"""
Extended Simple Adversary — simple_adversary_v3 https://pettingzoo.farama.org/_modules/pettingzoo/mpe/simple_adversary/simple_adversary/ to support multiple adversaries.

Changes from the original pettingzoo.mpe.simple_adversary: make_world accepts num_adversaries (default 1)
"""

import numpy as np
from gymnasium.utils import EzPickle

from pettingzoo.mpe._mpe_utils.core import Agent, Landmark, World
from pettingzoo.mpe._mpe_utils.scenario import BaseScenario
from pettingzoo.mpe._mpe_utils.simple_env import SimpleEnv, make_env
from pettingzoo.utils.conversions import parallel_wrapper_fn


class raw_env(SimpleEnv, EzPickle):
    def __init__(
        self,
        N=2,
        num_adversaries=1,
        max_cycles=25,
        continuous_actions=True,
        render_mode=None,
        dynamic_rescaling=False,
    ):
        EzPickle.__init__(
            self,
            N=N,
            num_adversaries=num_adversaries,
            max_cycles=max_cycles,
            continuous_actions=continuous_actions,
            render_mode=render_mode,
        )
        scenario = Scenario()
        world = scenario.make_world(N, num_adversaries)
        SimpleEnv.__init__(
            self,
            scenario=scenario,
            world=world,
            render_mode=render_mode,
            max_cycles=max_cycles,
            continuous_actions=continuous_actions,
            dynamic_rescaling=dynamic_rescaling,
        )
        self.metadata["name"] = "custom_simple_adversary_v0"


env = make_env(raw_env)
parallel_env = parallel_wrapper_fn(env)


class Scenario(BaseScenario):
    def make_world(self, N=2, num_adversaries=1):
        world = World()
        world.dim_c = 2

        num_agents = num_adversaries + N
        world.num_agents = num_agents

        # N landmarks regardless of adversary count (deception task)
        num_landmarks = N

        # agents
        world.agents = [Agent() for _ in range(num_agents)]
        for i in range(num_adversaries):
            world.agents[i].adversary = True
            world.agents[i].name = f"adversary_{i}"
            world.agents[i].collide = False
            world.agents[i].silent = True
            world.agents[i].size = 0.15

        for i in range(N):
            world.agents[num_adversaries + i].adversary = False
            world.agents[num_adversaries + i].name = f"agent_{i}"
            world.agents[num_adversaries + i].collide = False
            world.agents[num_adversaries + i].silent = True
            world.agents[num_adversaries + i].size = 0.15

        # landmarks
        world.landmarks = [Landmark() for _ in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = f"landmark {i}"
            landmark.collide = False
            landmark.movable = False
            landmark.size = 0.08

        return world

    def reset_world(self, world, np_random):
        # adversaries = red, good agents = blue
        for agent in world.agents:
            if agent.adversary:
                agent.color = np.array([0.85, 0.35, 0.35])
            else:
                agent.color = np.array([0.35, 0.35, 0.85])

        # landmarks default dark; target will be green
        for landmark in world.landmarks:
            landmark.color = np.array([0.15, 0.15, 0.15])

        # pick a random target landmark (same for all agents)
        goal = np_random.choice(world.landmarks)
        goal.color = np.array([0.15, 0.65, 0.15])
        for agent in world.agents:
            agent.goal_a = goal

        # random initial positions
        for agent in world.agents:
            agent.state.p_pos = np_random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for landmark in world.landmarks:
            landmark.state.p_pos = np_random.uniform(-1, +1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)

    def benchmark_data(self, agent, world):
        if agent.adversary:
            return np.sum(np.square(agent.state.p_pos - agent.goal_a.state.p_pos))
        dists = [
            np.sum(np.square(agent.state.p_pos - lm.state.p_pos))
            for lm in world.landmarks
        ]
        dists.append(
            np.sum(np.square(agent.state.p_pos - agent.goal_a.state.p_pos))
        )
        return tuple(dists)

    def good_agents(self, world):
        return [a for a in world.agents if not a.adversary]

    def adversaries(self, world):
        return [a for a in world.agents if a.adversary]

    def reward(self, agent, world):
        return (
            self.adversary_reward(agent, world)
            if agent.adversary
            else self.agent_reward(agent, world)
        )

    def agent_reward(self, agent, world):
        """
        Good agent reward:
          pos_rew     = -min distance from any good agent to the target  (always <= 0)
          adv_penalty = -min distance from any adversary to the target   (always <= 0)
          total       = pos_rew - adv_penalty
                      = -min_dist(good) + min_dist(adv)
        Positive when adversary is far (deception working), negative when adversary is close.
        """
        good_agents = self.good_agents(world)
        adversary_agents = self.adversaries(world)

        # negative component: penalise closest good agent's distance to target
        pos_rew = -min(
            np.sqrt(np.sum(np.square(a.state.p_pos - a.goal_a.state.p_pos)))
            for a in good_agents
        )

        # negative stored value: used as penalty via subtraction
        adv_penalty = -min(
            np.sqrt(np.sum(np.square(a.state.p_pos - a.goal_a.state.p_pos)))
            for a in adversary_agents
        )

        return pos_rew - adv_penalty

    def adversary_reward(self, agent, world):
        return -np.sqrt(
            np.sum(np.square(agent.state.p_pos - agent.goal_a.state.p_pos))
        )

    def observation(self, agent, world):
        # relative positions of all landmarks
        entity_pos = [
            entity.state.p_pos - agent.state.p_pos
            for entity in world.landmarks
        ]
        # relative positions of all other agents
        other_pos = [
            other.state.p_pos - agent.state.p_pos
            for other in world.agents
            if other is not agent
        ]

        if not agent.adversary:
            # good agents know their goal
            return np.concatenate(
                [agent.goal_a.state.p_pos - agent.state.p_pos] + entity_pos + other_pos
            )
        else:
            # adversary does NOT know which landmark is the target
            return np.concatenate(entity_pos + other_pos)
