from net.thinking import Thinking
import torch.nn as nn


class JepaThinking(nn.Module):
    def __init__(self, config_dict={}):
        super(JepaThinking, self).__init__()
        self.config_dict = config_dict

        self.output_dim = self.config_dict.pop("output_dim", 2)
        self.feal_dim = self.config_dict.get("feal_dim", 128)
        self.config_dict["output_dim"] = self.feal_dim

        self.feal = Thinking(config_dict)

        pred_dim = 64

        self.predictor = nn.Sequential(
            nn.Linear(self.feal_dim, pred_dim, bias=False),
            nn.BatchNorm1d(pred_dim),
            nn.ReLU(inplace=True),  # hidden layer
            nn.Linear(pred_dim, self.output_dim),
        )  # output layer

    def forward(self, x, x1):
        x = self.feal(x)
        x1 = self.feal(x1)
        return self.predictor(x), x1
