import numpy as np
import gymnasium as gym
import panda_gym
from gymnasium.spaces import Box, Dict
import time


class ImageObservationWrapper(gym.Wrapper):
    """
    包装器：将环境的渲染图像添加到观察空间中
    """

    def __init__(self, env, image_key="image"):
        super().__init__(env)

        # 确保环境使用rgb_array模式
        assert hasattr(env, "render_mode") and env.render_mode == "rgb_array"

        self.image_key = image_key

        # 获取一个示例图像来确定图像空间的形状
        self.env.reset()
        sample_image = self.env.render()

        if sample_image is None:
            raise ValueError("Environment render() returned None. Check render_mode.")

        # 观察空间，包含原始观察和图像
        self.observation_space = Dict(
            {
                self.image_key: Box(
                    low=0, high=255, shape=sample_image.shape, dtype=np.uint8
                ),
                **self.env.observation_space.spaces,
            }
        )

    def _get_obs_with_image(self, original_obs):
        """将图像添加到原始观察中"""
        image = self.env.render()
        combined_obs = original_obs.copy()
        combined_obs[self.image_key] = image
        return combined_obs

    def reset(self, **kwargs):
        """重置环境并返回包含图像的观察"""
        obs, info = self.env.reset(**kwargs)
        combined_obs = self._get_obs_with_image(obs)
        return combined_obs, info

    def step(self, action):
        """执行动作并返回包含图像的观察"""
        obs, reward, terminated, truncated, info = self.env.step(action)
        combined_obs = self._get_obs_with_image(obs)
        return combined_obs, reward, terminated, truncated, info


"""
https://panda-gym.readthedocs.io/en/latest/usage/environments.html
"""


def test():

    env_name = "PandaSlide-v3"
    env_name = "PandaPickAndPlaceDense-v3"

    # 创建原始环境
    base_env = gym.make(
        env_name,
        render_mode="rgb_array",
        renderer="OpenGL",
        render_width=480,
        render_height=480,
        render_target_position=[0, 0, 0],
        render_distance=0.8,
        render_yaw=135,
        render_pitch=-40,
        render_roll=0,
    )

    # 用包装器包装环境
    env = ImageObservationWrapper(base_env)

    print("=" * 40)

    print("Action space:", env.action_space)
    print("Observation space:", env.observation_space)
    print("=" * 40)

    observation, info = env.reset()
    print("Image shape:", observation["image"].shape)

    for _ in range(20):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)

        time.sleep(0.1)

        if terminated or truncated:
            observation, info = env.reset()

    env.close()


def test2():

    # env_name = "PandaSlide-v3"

    import gymnasium_robotics

    gym.register_envs(gymnasium_robotics)
    env_name = "FetchPickAndPlace-v4"

    # 创建原始环境
    # env = gym.make(env_name, render_mode="rgb_array")
    env = gym.make(env_name, render_mode="human")

    # 用包装器包装环境
    # env = ImageObservationWrapper(env)

    print()

    print("Action space:", env.action_space)
    print("Observation space:", env.observation_space)
    observation, info = env.reset()

    for _ in range(10):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)

        time.sleep(0.1)

        if terminated or truncated:
            observation, info = env.reset()

    env.close()


if __name__ == "__main__":
    # test2()
    test()
