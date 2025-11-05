import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import cv2


class BallEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        self.render_scale = 2

        # 桌面大小
        self.width = 256
        self.height = 256

        # 小球参数
        self.ball_radius = 3
        self.ball_pos = np.array([self.width // 2, self.height // 2], dtype=np.float32)
        self.ball_vel = np.zeros(2, dtype=np.float32)
        self.friction = 0.9  # 阻力系数

        # 目标点
        self.target_pos = np.array(
            [
                np.random.randint(10, self.width - 10),
                np.random.randint(10, self.height - 10),
            ],
            dtype=np.float32,
        )
        self.target_radius = 4

        # 动作空间：力在 x, y 方向，范围 -1 ~ 1
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

        # 观测空间：图像 64x64x3
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(self.height, self.width, 3), dtype=np.uint8
        )

        # Pygame 初始化
        if self.render_mode == "human":
            pygame.init()
            self.screen = pygame.display.set_mode(
                (self.width * self.render_scale, self.height * self.render_scale)
            )  # 放大 self.render_scale 倍显示
            self.clock = pygame.time.Clock()

    def reset(self, seed=None, options=None):
        self.ball_pos = np.array([self.width // 2, self.height // 2], dtype=np.float32)
        self.ball_vel = np.zeros(2, dtype=np.float32)
        self.target_pos = np.array(
            [
                np.random.randint(10, self.width - 10),
                np.random.randint(10, self.height - 10),
            ],
            dtype=np.float32,
        )
        return self._get_obs(), {}

    def step(self, action):
        action = np.clip(action, -1, 1)

        # 更新速度和位置
        self.ball_vel += action
        self.ball_vel *= self.friction
        self.ball_pos += self.ball_vel

        # 边界限制
        self.ball_pos = np.clip(self.ball_pos, 0, [self.width - 1, self.height - 1])

        # 计算奖励
        dist = np.linalg.norm(self.ball_pos - self.target_pos)
        done = dist < self.target_radius
        reward = 1.0 if done else -dist * 0.01  # 到达目标奖励1，其它按距离惩罚

        return self._get_obs(), reward, done, False, {}

    def _get_obs(self):
        # 创建RGB图像
        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        # 画目标点
        cv2.circle(
            img, tuple(self.target_pos.astype(int)), self.target_radius, (0, 255, 0), -1
        )
        # 画小球
        cv2.circle(
            img, tuple(self.ball_pos.astype(int)), self.ball_radius, (255, 0, 0), -1
        )
        return img

    def render(self):
        if self.render_mode == "human":
            surf = pygame.surfarray.make_surface(np.flip(self._get_obs(), axis=2))
            surf = pygame.transform.scale(
                surf, (self.width * self.render_scale, self.height * self.render_scale)
            )
            self.screen.blit(surf, (0, 0))
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.render_mode == "human":
            pygame.quit()


def test():
    env = BallEnv(render_mode="human")
    obs, _ = env.reset()

    done = False
    while not done:
        action = np.random.uniform(-1, 1, size=(2,))
        obs, reward, done, _, _ = env.step(action)
        env.render()

    env.close()


def test_keyborad():
    env = BallEnv(render_mode="human")

    print(f"obs: {env.observation_space}")
    print(f"action: {env.action_space}")

    obs, _ = env.reset()

    done = False
    action = np.zeros(2, dtype=np.float32)

    while True:
        # 事件处理
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.close()
                return

        keys = pygame.key.get_pressed()
        action[:] = 0  # 每次重置动作

        if keys[pygame.K_UP]:
            action[0] = -1
        if keys[pygame.K_DOWN]:
            action[0] = 1
        if keys[pygame.K_LEFT]:
            action[1] = -1
        if keys[pygame.K_RIGHT]:
            action[1] = 1

        # 执行动作
        obs, reward, done, _, _ = env.step(action)
        env.render()

        if done:
            print("到达目标！奖励:", reward)
            obs, _ = env.reset()


if __name__ == "__main__":
    test_keyborad()
