import gymnasium as gym
import numpy as np
import random

from ray.rllib.env.multi_agent_env import MultiAgentEnv
# from ray.rllib.utils import check_env


class DummyEnv(gym.Env):

    def __init__(self):
        super().__init__()
        self.num_agents = 5
        self.map_count = 10
        self._agent_ids = {agent for agent in range(self.num_agents)}

        self.observation_space = gym.spaces.Dict(
            {
                "NPC": gym.spaces.Box(-1, 1, shape=(self.num_agents, 20)),
                "Ego": gym.spaces.Box(-1, 1, shape=(1, 20)),
                "Map": gym.spaces.Box(-1, 1, shape=(self.map_count, 6))
            }
        )

        agent_action_space = gym.spaces.Box(-1, 1, shape=(2,))
        self.action_space = gym.spaces.Tuple((agent_action_space,) * self.num_agents)

    def reset(self, *, seed=None, options=None):
        self._step = 0
        obs = self.observation_space.sample()
        info = {}

        return obs, info

    def step(self, action_dict):
        self._step += 1

        obs = self.observation_space.sample()
        rewards = {agent: 10 for agent in range(self.num_agents)}
        terminated = self._step < 20
        truncated = False
        all_rewards = sum(rewards.values())
        info = {"rewards": rewards}

        return obs, all_rewards, terminated, truncated, info

"""
env = DummyEnv()
obs, info = env.reset()
print(obs[0]['Map'].shape)
print(env._agent_ids)
check_env(env)
"""
