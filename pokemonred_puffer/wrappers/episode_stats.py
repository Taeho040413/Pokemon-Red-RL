from typing import Any

import numpy as np
import gymnasium

import pufferlib.utils


class EpisodeStatsWrapper(gymnasium.Wrapper):
    def __init__(self, env, *args, **kwargs):
        super().__init__(env)
        self.reset()

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ):
        self.info = dict(episode_return=0, episode_length=0)
        return super().reset(seed=seed, options=options)

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)

        for k, v in pufferlib.utils.unroll_nested_dict(info):
            if "exploration_map" in k:
                self.info[k] = self.info.get(k, np.zeros_like(v)) + v
            elif "state" in k:
                if "state" not in self.info:
                    self.info["state"] = v
                elif isinstance(self.info["state"], dict) and isinstance(v, dict):
                    self.info["state"] |= v
                else:
                    try:
                        self.info["state"] |= v
                    except TypeError:
                        self.info["state"] = v
            else:
                self.info[k] = v

        # self.info['episode_return'].append(reward)
        self.info["episode_return"] += reward
        self.info["episode_length"] += 1

        info = {}
        lf = getattr(self.env, "log_frequency", None) or 0
        emit_periodic = bool(lf) and self.info["episode_length"] % lf == 0
        if terminated or truncated or emit_periodic:
            info = self.info

        return observation, reward, terminated, truncated, info
