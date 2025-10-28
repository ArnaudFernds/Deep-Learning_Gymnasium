import torch
import torch.nn as nn
import torch.nn.functional as F


class CnnPolicyValueCont(nn.Module):
    def __init__(self, action_dim=3):
        super().__init__()

        # === Feature extractor CNN ===
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        dummy = torch.zeros(1, 4, 96, 96)
        with torch.no_grad():
            feat = self.conv(dummy)
            self.flat_dim = feat.reshape(1, -1).size(1)

        # === Policy head (mean of Gaussian) ===
        self.fc_policy = nn.Sequential(
            nn.Linear(self.flat_dim, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
        )

        self.fc_logstd = nn.Parameter(torch.zeros(action_dim))

        # === Value head ===
        self.fc_value = nn.Sequential(
            nn.Linear(self.flat_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        feat = self.conv(x)
        feat = feat.reshape(feat.size(0), -1)

        mean   = self.fc_policy(feat)
        logstd = self.fc_logstd.expand_as(mean)
        value  = self.fc_value(feat)

        return mean, logstd, value

    def policy_forward(self, x):
        return self.forward(x)