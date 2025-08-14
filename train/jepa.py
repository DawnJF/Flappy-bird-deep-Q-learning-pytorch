import torch.nn as nn
from dataclasses import dataclass
from net.jepa_thinking import JepaThinking
from train.supervised import BaseModel


@dataclass
class Config:
    # 数据路径
    data_path: str = (
        "outputs/dataset_s4/observations_actions_flappy_bird_800000_20250806_003553.h5"
    )
    data_size: int = 20000

    # 训练参数
    batch_size: int = 64
    learning_rate: float = 1e-4
    num_step: int = 4000
    save_freq: int = num_step // 4
    logging_freq: int = 200

    # 模型参数
    action_dim: int = 2
    feal_dim: int = 0
    channel_dim: int = 1

    output_dir: str = "outputs/supervised"


class JepaThinkModelV1(BaseModel):
    """具体模型实现，封装了Thinking网络"""

    def __init__(self, config, device):
        super().__init__(config, device)
        self.action_dim = config["action_dim"]
        self.model = JepaThinking(config).to(device)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        assert x.shape[1] == 2

        p1 = self.model(x[:, 0])
        f1 = self.model.feat(p1)
        p2 = self.model(x[:, 1])
        p2 = p2.detach()
        f2 = self.model.feat(p2)

        action = p1[:, : -self.action_dim]
        return f1, f2, action

    def compute_loss(self, outputs, targets):
        feal_loss = self.criterion(outputs[0], outputs[1])
        action_loss = 0
        if targets is not None:
            action_loss = self.criterion(outputs[2], targets)
        return feal_loss + action_loss

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
