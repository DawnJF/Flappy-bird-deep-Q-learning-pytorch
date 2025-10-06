import gymnasium as gym
from gymnasium import spaces
import numpy as np
from envs.flappy_bird import FlappyBird


class FlappyBirdEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self):
        super().__init__()
        self.game = FlappyBird()
        # Observation: RGB image (H, W, C)
        obs_shape = (self.game.screen_width, self.game.screen_height, 3)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=obs_shape, dtype=np.uint8
        )
        # Action: 0 = do nothing, 1 = flap
        self.action_space = spaces.Discrete(2)
        self._last_obs = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.game.__init__()
        obs, _, _ = self.game.next_frame(0)
        self._last_obs = obs
        return obs, {}

    def step(self, action):
        obs, reward, terminated = self.game.next_frame(action)
        self._last_obs = obs
        truncated = False
        info = {}
        return obs, reward, terminated, truncated, info

    def render(self, mode="human"):
        # Already rendered by pygame, just return the last obs
        return self._last_obs

    def close(self):
        import pygame

        pygame.quit()
