import gymnasium as gym
import panda_gym
from envs.image_wrapper import ImageObservationWrapper


def get_env():
    """
    https://panda-gym.readthedocs.io/en/latest/usage/environments.html
    """

    env_name = "PandaPickAndPlaceDense-v3"

    # 创建原始环境
    env = gym.make(
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
    env = ImageObservationWrapper(env)

    return env
