from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..game_types import TOTAL_ACTIONS

class ResBlock(nn.Module):
    def __init__(self, num_channels: int):
        super().__init__()

        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return F.relu(out)

class PolicyValueNet(nn.Module):
    # Input 
    # (10, 5, 5) 5 for heights / 4 for workers / 1 for to move

    # Outputs
    # policy logits: 306
    # value: 1 between -1 and 1

    def __init__(self, num_channels: int = 64, num_res_blocks: int = 5):
        super().__init__()
        
        self.conv_in = nn.Conv2d(10, num_channels, kernel_size=3, padding=1, bias=False)
        self.bn_in = nn.BatchNorm2d(num_channels)

        self.res_blocks = nn.Sequential(
            *[ResBlock(num_channels) for _ in range(num_res_blocks)]
        )

        # Policy
        self.conv_p = nn.Conv2d(num_channels, 2, kernel_size=1)
        self.bn_p = nn.BatchNorm2d(2)
        self.fc_p = nn.Linear(2 * 5 * 5, TOTAL_ACTIONS)

        # Value
        self.conv_v = nn.Conv2d(num_channels, 1, kernel_size=1)
        self.bn_v = nn.BatchNorm2d(1)
        self.fc_v1 = nn.Linear(1 * 5 * 5, 32)
        self.fc_v2 = nn.Linear(32, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Body
        x = F.relu(self.bn_in(self.conv_in(x)))
        x = self.res_blocks(x)

        # Policy
        p = F.relu(self.bn_p(self.conv_p(x)))
        p = p.view(p.size(0), -1)
        policy_logits = self.fc_p(p)

        # Value
        v = F.relu(self.bn_v(self.conv_v(x)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.fc_v1(v))

        # Squash vlaue inbetween -1 and 1
        value = torch.tanh(self.fc_v2(v))

        return policy_logits, value