from src.net.thinking import Thinking
import torch.nn as nn


class JepaThinking(nn.Module):
    def __init__(self, config={}):
        super(JepaThinking, self).__init__()
        self.config = config

        self.output_dim = self.config.pop("output_dim", 2)
        self.feal_dim = self.config.get("feal_dim", 128)
        self.config["output_dim"] = self.feal_dim

        self.cognition = Thinking(self.config)

        pred_dim = 64

        self.forecast = nn.Sequential(
            nn.Linear(self.feal_dim, pred_dim, bias=False),
            nn.BatchNorm1d(pred_dim),
            nn.ReLU(inplace=True),  # hidden layer
            nn.Linear(pred_dim, self.feal_dim - self.output_dim),
        )  # output layer

    def forward(self, x):
        x = self.cognition(x)
        return x[:, -self.output_dim :]
