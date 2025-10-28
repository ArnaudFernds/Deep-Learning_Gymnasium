import numpy as np
import gymnasium as gym
from collections import deque


class FrameStack(gym.Wrapper):

    def __init__(self, env, k=4):
        super().__init__(env)
        self.k = k
        self.frames = deque(maxlen=k)

        h, w, c = env.observation_space.shape
        assert c == 1, "FrameStack attend des frames grayscale (C=1)"

        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(h, w, k),
            dtype=np.uint8,
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for _ in range(self.k):
            self.frames.append(obs)
        return self._get_obs(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self):
        return np.concatenate(list(self.frames), axis=-1)