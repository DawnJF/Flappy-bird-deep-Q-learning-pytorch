from dataclasses import dataclass
import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import tyro

sys.path.append(os.getcwd())
from src.obs_processor import ObsProcessor
from src.flappy_bird_env import FlappyBirdEnv
from src.net.thinking import Thinking
from src.utils import get_device


# Hyperparameters
@dataclass
class Config:
    learning_rate = 0.0002
    gamma = 0.98
    output_dir = "outputs"


def train_net(gamma, optimizer, data):
    R = 0
    optimizer.zero_grad()
    for r, prob in data[::-1]:
        R = r + gamma * R
        loss = -torch.log(prob) * R
        loss.backward()
    optimizer.step()


def main(config: Config):
    device = get_device()
    env = FlappyBirdEnv()
    pi = Thinking({"feal_dim": 0})
    pi.to(device)

    print(f"Observation space: {env.observation_space.shape}")

    state_processor = ObsProcessor(
        stack_size=4,
        original_image_size=env.observation_space.shape[:2],
        target_image_size=84,
        device=device,
    )

    score = 0.0
    print_interval = 20

    data = []
    optimizer = optim.Adam(pi.parameters(), lr=config.learning_rate)

    for n_epi in range(10000):
        s, _ = env.reset()
        s = state_processor.initialize_state(s)
        done = False

        print(f"Episode {n_epi + 1} starting...")

        while not done:

            action = pi(s)

            prob = F.softmax(action, dim=-1)
            m = Categorical(prob)
            a = m.sample()

            print(
                f"Action: {a.item()}, Probabilities: ({prob[0, 0]:.2f}, {prob[0, 1]:.2f})"
            )

            s_prime, r, done, truncated, info = env.step(a.item())

            data.append((r, prob[0, a]))

            s = state_processor.update_state(s_prime)
            score += r

        train_net(config.gamma, optimizer, data)
        data = []

        if n_epi % print_interval == 0 and n_epi != 0:
            print(
                "# of episode :{}, avg score : {}".format(n_epi, score / print_interval)
            )
            score = 0.0
    env.close()


if __name__ == "__main__":
    config = tyro.cli(Config)

    main(config)
