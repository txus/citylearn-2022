from numbers import Number
from typing import Dict, List

import numpy as np
import supersuit as ss
from citylearn.citylearn_pettingzoo import CityLearnPettingZooEnv
from supersuit.vector.sb3_vector_wrapper import SB3VecEnvWrapper

from rewards.user_reward import UserReward


def to_numpy(values_by_agent: Dict[str, List[Number]]):
    return {k: np.array(v) for k, v in values_by_agent.items()}


class Env(CityLearnPettingZooEnv):
    # The signature is outdated and undocumented, so we monkeypatch it
    # We also need to return a Numpy array instead of plain lists,
    # otherwise SB3 will yell at us
    def reset(self, seed=0, return_info=True, options={}):
        return to_numpy(super().reset(seed=seed))

    def step(self, actions):
        observations, rewards, dones, infos = super().step(actions)
        return to_numpy(observations), to_numpy(rewards), dones, infos


def make_environment(schema_path: str) -> SB3VecEnvWrapper:
    env = Env(schema=schema_path)
    env = ss.pad_observations_v0(env)
    env = ss.pad_action_space_v0(env)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    return ss.concat_vec_envs_v1(env, 16, num_cpus=8, base_class="stable_baselines3")
