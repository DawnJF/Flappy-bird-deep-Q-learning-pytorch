"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""

import torch.nn as nn

from src.thinking import Thinking


class Net1(nn.Module):
    def __init__(self):
        super(Net1, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4), nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(inplace=True)
        )

        self.fc1 = nn.Sequential(nn.Linear(7 * 7 * 64, 512), nn.ReLU(inplace=True))
        self.fc2 = nn.Linear(512, 2)
        self._create_weights()

    def _create_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, -0.01, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, input):  # (1, 4, 84, 84)
        output = self.conv1(input)
        output = self.conv2(output)  # (1, 32, 20, 20)
        output = self.conv3(output)  # (1, 64, 9, 9)
        output = output.view(output.size(0), -1)  # (1, 64, 7, 7)
        output = self.fc1(output)  # (1, 3136)
        output = self.fc2(output)  # (1, 512)

        return output  # (1, 2)


class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4), nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(inplace=True)
        )

        self.fc1 = nn.Sequential(nn.Linear(7 * 7 * 64, 1024), nn.ReLU(inplace=True))
        self.fc1_1 = nn.Sequential(nn.Linear(1024, 512), nn.ReLU(inplace=True))
        self.fc2 = nn.Linear(512, 2)
        self._create_weights()

    def _create_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, -0.01, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, input):  # (1, 4, 84, 84)
        output = self.conv1(input)
        output = self.conv2(output)  # (1, 32, 20, 20)
        output = self.conv3(output)  # (1, 64, 9, 9)
        output = output.view(output.size(0), -1)  # (1, 64, 7, 7)
        output = self.fc1(output)  # (1, 3136)
        output = self.fc1_1(output)  # (1, 512)
        output = self.fc2(output)  # (1, 512)

        return output  # (1, 2)


class DeepQNetwork(Thinking):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return super().forward(x)[0]
