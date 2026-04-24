"""
network_TD3.py — TD3 Specific Actor / Critic Network Definitions
================================================================

Dimension conventions (Consistent with common/env.py):
    SCALAR_DIM     = 27    Actor scalar input
    PRIVILEGED_DIM = 44    Critic pure scalar input (Full state, excluding CNN)
    ACTION_DIM     = 5     Output action dimension
    IMG_H, IMG_W   = 64, 64

Core TD3 changes:
    1. Actor: Deterministic Policy, directly outputs Tanh actions without log_std.
    2. Critic: Twin Critics, input is not just state but also concatenated action (priv_obs + action) -> Q(s, a).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# -- Note: Path modified to import from common --------------------------------
from common.env import (
    SCALAR_DIM,
    PRIVILEGED_DIM,
    ACTION_DIM,
    IMG_H, IMG_W,
)

# Visual feature dimension output by each CNN Encoder
VISUAL_FEAT_DIM = 32

# Actor concat dimension: scalar(27) + wrist_feat(32) + global_feat(32)
ACTOR_INPUT_DIM = SCALAR_DIM + VISUAL_FEAT_DIM * 2           # 91
# Critic independent input dimension: Pure scalar privileged obs(44) + action(5) = 49
CRITIC_INPUT_DIM = PRIVILEGED_DIM + ACTION_DIM                # 49


# -----------------------------------------------------------------------------
# Depth Map CNN Encoder (Separate instances for wrist and global cameras)
# -----------------------------------------------------------------------------
class DepthEncoder(nn.Module):
    def __init__(self, out_dim: int = VISUAL_FEAT_DIM):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.ELU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ELU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ELU(inplace=True),
        )
        conv_out_dim = 64 * (IMG_H // 8) * (IMG_W // 8)
        self.proj = nn.Sequential(
            nn.Linear(conv_out_dim, out_dim),
            nn.ELU(inplace=True),
        )

    def forward(self, depth: torch.Tensor) -> torch.Tensor:
        x = self.conv(depth)
        x = x.flatten(start_dim=1)
        return self.proj(x)


# -----------------------------------------------------------------------------
# TD3 Actor Network (Outputs deterministic actions)
# -----------------------------------------------------------------------------
class TD3_Actor(nn.Module):
    def __init__(self, max_action: float = 1.0, hidden_dim: int = 256):
        super().__init__()
        self.max_action = max_action

        # -- Visual Encoders --------------------------------------------------
        self.wrist_encoder = DepthEncoder(VISUAL_FEAT_DIM)
        self.global_encoder = DepthEncoder(VISUAL_FEAT_DIM)

        # -- Scalar + Visual Feature Fusion MLP (TD3 outputs deterministic actions)
        self.mlp = nn.Sequential(
            nn.Linear(ACTOR_INPUT_DIM, hidden_dim),
            nn.ELU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ELU(inplace=True),
            nn.Linear(hidden_dim // 2, ACTION_DIM),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.zeros_(m.bias)
        last_linear = [m for m in self.mlp if isinstance(m, nn.Linear)][-1]
        nn.init.orthogonal_(last_linear.weight, gain=0.01)

    def encode(self, scalar_obs, wrist_depth, global_depth):
        wrist_feat = self.wrist_encoder(wrist_depth)
        global_feat = self.global_encoder(global_depth)
        return torch.cat([scalar_obs, wrist_feat, global_feat], dim=1)

    def forward(self, scalar_obs, wrist_depth, global_depth):
        """
        TD3 Core: Directly output deterministic actions multiplied by max_action.
        No distribution sampling involved.
        """
        fused = self.encode(scalar_obs, wrist_depth, global_depth)
        a = self.mlp(fused)
        return self.max_action * torch.tanh(a)


# -----------------------------------------------------------------------------
# TD3 Critic Network (Twin Critics / Double Q-Network)
# -----------------------------------------------------------------------------
class TD3_Critic(nn.Module):
    def __init__(self, hidden_dim: int = 256):
        super().__init__()

        # -- Q1 Network -------------------------------------------------------
        self.q1_mlp = nn.Sequential(
            # 44(state) + 5(action) = 49
            nn.Linear(CRITIC_INPUT_DIM, hidden_dim),
            nn.ELU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ELU(inplace=True),
            nn.Linear(hidden_dim // 2, 1),
        )

        # -- Q2 Network -------------------------------------------------------
        self.q2_mlp = nn.Sequential(
            nn.Linear(CRITIC_INPUT_DIM, hidden_dim),
            nn.ELU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ELU(inplace=True),
            nn.Linear(hidden_dim // 2, 1),
        )

        self._init_weights(self.q1_mlp)
        self._init_weights(self.q2_mlp)

    def _init_weights(self, mlp_net):
        for m in mlp_net:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.zeros_(m.bias)
        last_linear = [m for m in mlp_net if isinstance(m, nn.Linear)][-1]
        nn.init.orthogonal_(last_linear.weight, gain=0.01)

    def forward(self, critic_obs: torch.Tensor, action: torch.Tensor):
        """
        TD3 requires calculating both Q1 and Q2 to compute the Critic Loss.
        """
        sa = torch.cat([critic_obs, action], dim=1)
        q1 = self.q1_mlp(sa)
        q2 = self.q2_mlp(sa)
        return q1, q2

    def Q1(self, critic_obs: torch.Tensor, action: torch.Tensor):
        """
        When updating the Actor, only Q1 calculation 
        and ascent are required.
        """
        sa = torch.cat([critic_obs, action], dim=1)
        q1 = self.q1_mlp(sa)
        return q1
