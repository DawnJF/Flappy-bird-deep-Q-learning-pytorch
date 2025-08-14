import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from dataclasses import dataclass
import sys
import os
import tyro

sys.path.append(os.getcwd())
from src.net.jepa_thinking import JepaThinking
from train import supervised
from train.supervised import BaseModel, SupervisedTrainer


class JepaThinkModelV1(BaseModel):
    """具体模型实现，封装了Thinking网络"""

    def __init__(self, config, device):
        super().__init__(config, device)
        self.action_dim = config["output_dim"]
        self.model = JepaThinking(config).to(device)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, batch):
        x = batch[0]
        assert x.shape[1] == 2

        p1 = self.model.feal(x[:, 0])
        f1 = self.model.predictor(p1)

        p2 = self.model.feal(x[:, 1])
        p2 = p2.detach()
        f2 = self.model.predictor(p2)

        action = p1[:, -self.action_dim :]
        return (
            action,
            f1,
            f2,
        )

    def compute_loss(self, outputs, batch):
        feal_loss = self.criterion(outputs[1], outputs[2])
        mask = batch[1]

        outputs_masked = outputs[0][mask]  # [N_mask, C]
        labels_masked = batch[-1][mask]  # [N_mask]
        action_mask_loss = self.criterion(outputs_masked, labels_masked)

        return feal_loss + action_mask_loss

    def parameters(self):
        return self.model.parameters()

    def train(self, mode=True):
        self.model.train(mode)

    def eval(self):
        self.model.eval()

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)

    def state_dict(self):
        return self.model.state_dict()


class FlappyBirdJepaDataset(Dataset):
    """
    9/10的数据去掉action
    每个item包含当前obs和next obs
    """

    def __init__(self, observations, actions):
        self.data = []
        length = len(observations) - 1

        keep_actions = np.zeros(length, dtype=bool)
        # 随机选择 num_true 个索引置为 True
        true_indices = np.random.choice(length, size=length // 10, replace=False)
        keep_actions[true_indices] = True

        for i in range(length):
            action = actions[i]
            obs = np.expand_dims(observations[i], axis=0)
            next_obs = np.expand_dims(observations[i + 1], axis=0)
            state = np.concatenate([obs, next_obs])

            mask = keep_actions[i]

            self.data.append((state, mask, action))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        obs, mask, action = self.data[idx]
        return (
            torch.tensor(obs, dtype=torch.float32),
            torch.tensor(mask, dtype=torch.bool),
            torch.tensor(action, dtype=torch.long),
        )


@dataclass
class Config(supervised.Config):
    model_class = JepaThinkModelV1
    dataset_class = FlappyBirdJepaDataset

    output_dir: str = "outputs/jepa_v1"


def train(config: Config):
    """训练模型"""
    trainer = SupervisedTrainer(
        config, model_class=JepaThinkModelV1, dataset_class=FlappyBirdJepaDataset
    )
    trainer.train()


if __name__ == "__main__":
    # 使用tyro解析命令行参数
    config = tyro.cli(Config)

    train(config)
