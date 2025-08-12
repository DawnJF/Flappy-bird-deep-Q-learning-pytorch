import os
import shutil
from random import random, randint, sample
import sys
import numpy as np
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from dataclasses import dataclass
import tyro

sys.path.append(os.getcwd())
from src.net.deep_q_network import DeepQNetwork
from src.flappy_bird_env import FlappyBirdEnv
from src.obs_processor import ObsProcessor
from src.utils import get_device


@dataclass
class TrainingConfig:
    """Configuration for Deep Q Network training to play Flappy Bird"""

    image_size: int = 84
    """The common width and height for all images"""

    batch_size: int = 24
    """The number of images per batch"""

    optimizer: str = "adam"
    """Optimizer choice: sgd or adam"""

    lr: float = 1e-6

    gamma: float = 0.99
    """Discount factor"""

    initial_epsilon: float = 0.2
    """Initial epsilon for exploration"""

    final_epsilon: float = 1e-4
    """Final epsilon for exploration"""

    num_iters: int = 1000000
    """Number of training iterations"""

    replay_memory_size: int = 5000
    """Number of experiences to store in replay memory"""

    output_path: str = "outputs/dqn"
    """Path for tensorboard logs"""


def train(opt: TrainingConfig):

    device = get_device()

    model = DeepQNetwork()
    target_model = DeepQNetwork().to(device)  # 定义目标网络
    target_model.load_state_dict(model.state_dict())  # 初始化目标网络

    if os.path.isdir(opt.output_path):
        shutil.rmtree(opt.output_path)
    os.makedirs(opt.output_path)

    writer = SummaryWriter(opt.output_path)

    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    criterion = nn.MSELoss()

    env = FlappyBirdEnv()
    obs, _ = env.reset()

    state_processor = ObsProcessor(
        stack_size=4,
        original_image_size=obs.shape[:2],
        target_image_size=opt.image_size,
        device=device,
    )

    model = model.to(device)

    # 初始化状态
    state = state_processor.initialize_state(obs)

    replay_memory = []
    iter = 0
    episode_steps = 0  # 记录当前游戏的步数
    target_update_freq = 1000  # 每隔 1000 步更新一次目标网络

    while iter < opt.num_iters:
        prediction = model(state)[0]
        print("Prediction: ", prediction)

        # Exploration or exploitation
        epsilon = opt.final_epsilon + (
            (opt.num_iters - iter)
            * (opt.initial_epsilon - opt.final_epsilon)
            / opt.num_iters
        )
        u = random()
        random_action = u <= epsilon
        if random_action:
            print("Perform a random action")
            # action = 1 if random() <= 0.1 else 0
            action = randint(0, 1)
        else:
            action = torch.argmax(prediction).item()

        next_obs, reward, terminal, truncated, info = env.step(action)
        next_state = state_processor.update_state(next_obs)
        episode_steps += 1  # 增加步数计数

        replay_memory.append([state, action, reward, next_state, terminal])
        if len(replay_memory) > opt.replay_memory_size:
            del replay_memory[0]

        batch = sample(replay_memory, min(len(replay_memory), opt.batch_size))
        state_batch, action_batch, reward_batch, next_state_batch, terminal_batch = zip(
            *batch
        )

        state_batch = torch.cat(tuple(state for state in state_batch))
        action_batch = torch.from_numpy(
            np.array(
                [[1, 0] if action == 0 else [0, 1] for action in action_batch],
                dtype=np.float32,
            )
        )
        reward_batch = torch.from_numpy(
            np.array(reward_batch, dtype=np.float32)[:, None]
        )
        next_state_batch = torch.cat(tuple(state for state in next_state_batch))

        state_batch = state_batch.to(device)
        action_batch = action_batch.to(device)
        reward_batch = reward_batch.to(device)
        next_state_batch = next_state_batch.to(device)

        current_prediction_batch = model(state_batch)
        with torch.no_grad():  # 不需要梯度
            next_prediction_batch = target_model(next_state_batch)  # 使用目标网络
        # next_prediction_batch = model(next_state_batch)

        y_batch = torch.cat(
            tuple(
                reward if terminal else reward + opt.gamma * torch.max(prediction)
                for reward, terminal, prediction in zip(
                    reward_batch, terminal_batch, next_prediction_batch
                )
            )
        )

        q_value = torch.sum(current_prediction_batch * action_batch, dim=1)

        optimizer.zero_grad()
        loss = criterion(q_value, y_batch)
        loss.backward()
        optimizer.step()

        # 定期更新目标网络
        if iter % target_update_freq == 0:
            target_model.load_state_dict(model.state_dict())

        # 如果游戏结束，重置环境
        if terminal or truncated:
            print(f"Episode ended after {episode_steps} steps")
            writer.add_scalar("Train/Episode_Steps", episode_steps, iter)
            episode_steps = 0  # 重置步数计数
            obs, _ = env.reset()
            state = state_processor.initialize_state(obs)
        else:
            state = next_state

        iter += 1
        print(
            "Iteration: {}/{}, Action: {}, Loss: {}, Epsilon {}, Reward: {}, Q-value: {}".format(
                iter + 1,
                opt.num_iters,
                action,
                loss,
                epsilon,
                reward,
                torch.max(prediction),
            )
        )
        writer.add_scalar("Train/Loss", loss, iter)
        writer.add_scalar("Train/Epsilon", epsilon, iter)
        writer.add_scalar("Train/Reward", reward, iter)
        writer.add_scalar("Train/Q-value", torch.max(prediction), iter)
        if (iter + 1) % 200000 == 0:
            torch.save(model, "{}/flappy_bird_{}".format(opt.output_path, iter + 1))
    torch.save(model, "{}/flappy_bird".format(opt.output_path))
    env.close()


if __name__ == "__main__":
    opt = tyro.cli(TrainingConfig)
    train(opt)
