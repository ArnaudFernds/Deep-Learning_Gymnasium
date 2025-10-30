import torch
import torch.nn as nn
import torch.nn.functional as F

# ================================================================
# CnnPolicyValueCont
# ------------------------------------------------
# This class defines the neural network used by the PPO agent
# for continuous control in the CarRacing-v3 environment.
#
# The network is composed of:
#   1. A convolutional feature extractor (shared backbone)
#   2. A policy head (actor): outputs mean of Gaussian actions
#   3. A value head (critic): estimates the state value V(s)
#
# Inputs:  4 stacked grayscale frames (4 x 96 x 96)
# Outputs: action distribution parameters (mean, logstd)
#           and a scalar value estimate.
# ================================================================
class CnnPolicyValueCont(nn.Module):
    def __init__(self, action_dim=3):
        super().__init__()

        # === 1. Feature extractor (Convolutional Neural Network) ===
        # Extracts spatial features from the stacked grayscale frames.
        # The architecture progressively downsamples the input and
        # captures motion and geometry from visual data.
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),  # (4 input frames → 32 feature maps)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), # (32 → 64 channels)
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), # Final conv layer (64 feature maps)
            nn.ReLU(),
        )

        # Dummy input to automatically compute flattened size
        dummy = torch.zeros(1, 4, 96, 96)
        with torch.no_grad():
            feat = self.conv(dummy)
            self.flat_dim = feat.reshape(1, -1).size(1)  # number of features after flattening

        # === 2. Policy head (Actor) ===
        # Outputs the mean vector of a Gaussian distribution over actions.
        # Each action dimension (steer, gas, brake) will have a mean value.
        self.fc_policy = nn.Sequential(
            nn.Linear(self.flat_dim, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),  # Output: mean for each action dimension
        )

        # Log standard deviation parameter (learnable per action)
        self.fc_logstd = nn.Parameter(torch.zeros(action_dim))

        # === 3. Value head (Critic) ===
        # Outputs a scalar value estimating V(s) = expected return from this state.
        self.fc_value = nn.Sequential(
            nn.Linear(self.flat_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    # =============================================================
    # FORWARD PASS
    # -------------------------------------------------
    # Takes a batch of images, extracts features,
    # and returns the policy (mean, logstd) and value outputs.
    # =============================================================
    def forward(self, x):
        # CNN feature extraction
        feat = self.conv(x)
        feat = feat.reshape(feat.size(0), -1)

        # Actor: output Gaussian mean (continuous control)
        mean = self.fc_policy(feat)

        # Expand log standard deviation to same shape as mean
        logstd = self.fc_logstd.expand_as(mean)

        # Critic: value estimation
        value = self.fc_value(feat)

        # Return all outputs
        return mean, logstd, value

    # Alias for policy-only forward (optional, for clarity)
    def policy_forward(self, x):
        return self.forward(x)
