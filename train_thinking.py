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

checkpoint_freq = 20000
max_steps = 216000


class ThinkingTrainer:
    def __init__(self, opt):
        self.output_path = opt.log_path
        self.device = get_device()
        self.model = Thinking().to(self.device)

        self.writer = SummaryWriter(self.output_path)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=opt.lr)
        self.criterion = nn.MSELoss()
        self.think_criterion = nn.CosineSimilarity(dim=1)

        self.last_embedding = None
        self.iter = 0

    def step(self, state, action):
        state = state.to(self.device)
        action = torch.tensor([action]).to(self.device)
        action_batch = torch.from_numpy(
            np.array(
                [[1, 0] if action == 0 else [0, 1]],
                dtype=np.float32,
            )
        )
        action_batch = action_batch.to(self.device)

        pred_act, embedding = self.model(state, action)

        if self.last_embedding is not None:
            self.iter += 1
            iter = self.iter

            self.optimizer.zero_grad()
            thinking_loss = (
                1 - self.think_criterion(embedding, self.last_embedding).mean()
            ) * 0.001

            mse_loss = self.criterion(pred_act, action_batch)

            loss = mse_loss + thinking_loss

            loss.backward()
            self.optimizer.step()

            print(
                "Iter: {}, Loss: {}, MSE Loss: {}, Thinking Loss: {}".format(
                    iter, loss, mse_loss, thinking_loss
                )
            )

            self.writer.add_scalar("Train/Loss", loss, iter)
            self.writer.add_scalar("Train/MSE Loss", mse_loss, iter)
            self.writer.add_scalar("Train/Thinking Loss", thinking_loss, iter)

            if iter % checkpoint_freq == 0:
                torch.save(
                    self.model, "{}/flappy_bird_{}".format(self.output_path, iter + 1)
                )

        self.last_embedding = embedding.detach()


def train(opt):

    trainer = ThinkingTrainer(opt)

    model = torch.load(
        "trained_models/net_2/flappy_bird_1000000",
        map_location=lambda storage, loc: storage,
    )
    model.eval()
    game_state = FlappyBird()
    image, reward, terminal = game_state.next_frame(0)
    image = pre_processing(
        image[: game_state.screen_width, : int(game_state.base_y)],
        opt.image_size,
        opt.image_size,
    )
    image = torch.from_numpy(image)
    if torch.cuda.is_available():
        model.cuda()
        image = image.cuda()
    state = torch.cat(tuple(image for _ in range(4)))[None, :, :, :]

    iter = -5
    while iter < max_steps:
        prediction = model(state)[0]
        print(prediction.shape)
        action = torch.argmax(prediction).item()

        trainer.step(state, action)

        next_image, reward, terminal = game_state.next_frame(action)
        next_image = pre_processing(
            next_image[: game_state.screen_width, : int(game_state.base_y)],
            opt.image_size,
            opt.image_size,
        )
        next_image = torch.from_numpy(next_image)
        if torch.cuda.is_available():
            next_image = next_image.cuda()
        next_state = torch.cat((state[0, 1:, :, :], next_image))[None, :, :, :]

        state = next_state
        iter += 1


if __name__ == "__main__":
    opt = get_args("test_thinking")
    train(opt)
