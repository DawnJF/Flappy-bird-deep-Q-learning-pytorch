"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""

import os
import shutil
from random import random, randint, sample

import numpy as np
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter

from src.thinking import Thinking
from src.flappy_bird import FlappyBird
from src.utils import pre_processing, get_args, get_device

device = get_device()


def train(opt):
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)
    model = Thinking()
    target_model = Thinking().to(device)  # 定义目标网络
    target_model.load_state_dict(model.state_dict())  # 初始化目标网络

    if os.path.isdir(opt.log_path):
        shutil.rmtree(opt.log_path)
    os.makedirs(opt.log_path)
    writer = SummaryWriter(opt.log_path)

    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    criterion = nn.MSELoss()
    think_criterion = nn.CosineSimilarity(dim=1)
    game_state = FlappyBird()
    image, reward, terminal = game_state.next_frame(0)
    image = pre_processing(
        image[: game_state.screen_width, : int(game_state.base_y)],
        opt.image_size,
        opt.image_size,
    )
    image = torch.from_numpy(image)

    model = model.to(device)
    image = image.to(device)

    # 在深度强化学习中，通常使用一个序列帧来描述环境的状态，帮助模型捕捉动态变化。这里假设一开始所有帧都是相同的。
    # image: (84, 84)
    # state: (1, 4, 84, 84)
    state = torch.cat(tuple(image for _ in range(4)))[None, :, :, :]

    replay_memory = []
    iter = 0
    target_update_freq = 1000  # 每隔 1000 步更新一次目标网络

    while iter < opt.num_iters:
        prediction, _ = model(state)
        prediction = prediction[0]
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
            if random() <= 0.1:
                action = 1
            else:
                action = 0
            # action = randint(0, 1)
        else:

            action = torch.argmax(prediction).item()

        next_image, reward, terminal = game_state.next_frame(action)
        next_image = pre_processing(
            next_image[: game_state.screen_width, : int(game_state.base_y)],
            opt.image_size,
            opt.image_size,
        )
        next_image = torch.from_numpy(next_image)

        next_image = next_image.to(device)
        next_state = torch.cat((state[0, 1:, :, :], next_image))[None, :, :, :]

        replay_memory.append([state, action, reward, next_state, terminal])
        if len(replay_memory) > opt.replay_memory_size:
            del replay_memory[0]

        batch = sample(replay_memory, min(len(replay_memory), opt.batch_size))
        state_batch, action_batch, reward_batch, next_state_batch, terminal_batch = zip(
            *batch
        )

        state_batch = torch.cat(tuple(state for state in state_batch))
        action_batch = torch.tensor(action_batch)
        reward_batch = torch.from_numpy(
            np.array(reward_batch, dtype=np.float32)[:, None]
        )
        next_state_batch = torch.cat(tuple(state for state in next_state_batch))

        state_batch = state_batch.to(device)
        action_batch = action_batch.to(device)
        reward_batch = reward_batch.to(device)
        next_state_batch = next_state_batch.to(device)

        current_prediction_batch, predict = model(state_batch, action_batch)

        with torch.no_grad():  # 不需要梯度
            next_prediction_batch, embedding = target_model(
                next_state_batch
            )  # 使用目标网络

        y_batch = torch.cat(
            tuple(
                reward if terminal else reward + opt.gamma * torch.max(prediction)
                for reward, terminal, prediction in zip(
                    reward_batch, terminal_batch, next_prediction_batch
                )
            )
        )

        action_batch = torch.from_numpy(
            np.array(
                [[1, 0] if action == 0 else [0, 1] for action in action_batch],
                dtype=np.float32,
            )
        )
        action_batch = action_batch.to(device)
        q_value = torch.sum(current_prediction_batch * action_batch, dim=1)

        optimizer.zero_grad()
        thinking_loss = 1 - think_criterion(embedding, predict).mean()
        mse_loss = criterion(q_value, y_batch)

        loss = mse_loss + thinking_loss

        loss.backward()
        optimizer.step()

        # 定期更新目标网络
        if iter % target_update_freq == 0:
            target_model.load_state_dict(model.state_dict())

        state = next_state
        iter += 1
        print(
            "Iteration: {}/{}, Action: {}, Loss: {}({}, {}), Epsilon {}, Reward: {}, Q-value: {}".format(
                iter + 1,
                opt.num_iters,
                action,
                mse_loss,
                thinking_loss,
                loss,
                epsilon,
                reward,
                torch.max(prediction),
            )
        )
        writer.add_scalar("Train/Loss", loss, iter)
        writer.add_scalar("Train/MSE Loss", mse_loss, iter)
        writer.add_scalar("Train/Thinking Loss", thinking_loss, iter)
        writer.add_scalar("Train/Epsilon", epsilon, iter)
        writer.add_scalar("Train/Reward", reward, iter)
        writer.add_scalar("Train/Q-value", torch.max(prediction), iter)
        if (iter + 1) % 200000 == 0:
            torch.save(model, "{}/flappy_bird_{}".format(opt.saved_path, iter + 1))
    torch.save(model, "{}/flappy_bird".format(opt.saved_path))


if __name__ == "__main__":
    opt = get_args()
    train(opt)
