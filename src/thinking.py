"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""

import torch
import torch.nn as nn


class Thinking(nn.Module):
    def __init__(self):
        super(Thinking, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4), nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(inplace=True)
        )

        self.fc1 = nn.Sequential(
            nn.Linear(7 * 7 * 64 + 1, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 64),
            nn.LayerNorm(64),
        )
        self.fc2 = nn.Linear(64, 2)
        # self._create_weights()

    def _create_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, -0.01, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, input, action=None):
        output = self.conv1(input)
        output = self.conv2(output)
        output = self.conv3(output)
        output = output.view(output.size(0), -1)

        if action is None:
            action = torch.full((output.shape[0],), -1).to(output.device)
        action = action.unsqueeze(1)

        output = torch.cat((output, action), dim=1)

        embedding = self.fc1(output)
        output = self.fc2(embedding)

        return output, embedding
