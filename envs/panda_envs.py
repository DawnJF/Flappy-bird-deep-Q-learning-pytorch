import time
import gymnasium as gym
import panda_gym
import os
import sys

sys.path.append(os.getcwd())
from envs.image_wrapper import ImageObservationWrapper


def get_env(env_name="PandaPickAndPlaceDense-v3"):
    """
    https://panda-gym.readthedocs.io/en/latest/usage/environments.html
    """

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


def manual_control():
    # env = gym.make("PandaReach-v3", render_mode="human")
    env = get_env("PandaReach-v3")
    print("=" * 40)
    print("Action space:", env.action_space)
    print("Observation space:", env.observation_space)
    print("=" * 40)
    observation, info = env.reset()

    for _ in range(1000):
        current_position = observation["observation"][0:3]
        desired_position = observation["desired_goal"][0:3]
        action = 5.0 * (desired_position - current_position)
        print("-" * 40)
        print("Current position:", current_position)
        print("Desired position:", desired_position)
        print("Action:", action)

        observation, reward, terminated, truncated, info = env.step(action)
        time.sleep(0.5)

        if terminated or truncated:
            observation, info = env.reset()
            break

    env.close()


if __name__ == "__main__":
    manual_control()
