import numpy as np
import gymnasium as gym
import cv2


class GrayScaleWrapper(gym.ObservationWrapper):

    def __init__(self, env):
        super().__init__(env)

        h, w, _ = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(h, w, 1),
            dtype=np.uint8,
        )

    def observation(self, obs):
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        gray = np.expand_dims(gray, axis=-1)
        return gray