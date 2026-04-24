"""
network.py — Actor / Critic Network Definitions
===============================================

Dimension Conventions (consistent with env.py):
    SCALAR_DIM     = 27    Actor scalar input
    PRIVILEGED_DIM = 44    Critic pure scalar input (full state, no CNN)
    ACTION_DIM     = 5     Output action dimension
    IMG_H, IMG_W   = 64, 64

Actor Input Concatenation:
    scalar(27) + wrist_feat(32) + global_feat(32) = 91 dimensions
    CNN is fully enabled from Stage 1 onwards (no alpha warm-up mechanism)

Critic Input:
    critic_obs(44) — Pure scalar privileged observation, fully decoupled from Actor
    Critic has no CNN and does not depend on actor.encode()
"""

import torch
import torch.nn as nn
from torch.distributions import Normal

# -- Import dimension constants from env.py to maintain a single source of truth --
from common.env import (
    SCALAR_DIM,
    PRIVILEGED_DIM,
    ACTION_DIM,
    IMG_H, IMG_W,
)

# Output visual feature dimension for each CNN Encoder
VISUAL_FEAT_DIM = 32

# Actor concat dimension: scalar(27) + wrist_feat(32) + global_feat(32)
ACTOR_INPUT_DIM = SCALAR_DIM + VISUAL_FEAT_DIM * 2           # 91
# Critic independent input dimension: pure scalar privileged obs, no actor_fused
CRITIC_INPUT_DIM = PRIVILEGED_DIM                              # 44


# -----------------------------------------------------------------------------
# Depth Map CNN Encoder (Independent instances for wrist and global views)
# -----------------------------------------------------------------------------
class DepthEncoder(nn.Module):
    """
    Encodes single-channel depth maps (B, 1, 64, 64) into VISUAL_FEAT_DIM feature vectors.

    Structure: 3 × Conv(stride=2) → Flatten → Linear → ReLU
    Spatial size transition: 64 → 32 → 16 → 8, final feature map is 64×8×8 = 4096 dims.
    """

    def __init__(self, out_dim: int = VISUAL_FEAT_DIM, aux_dim: int = 0):
        super().__init__()
        self.aux_dim = aux_dim
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2,
                      padding=1),  # → (B,16,32,32)
            nn.ELU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2,
                      padding=1),  # → (B,32,16,16)
            nn.ELU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2,
                      padding=1),  # → (B,64, 8, 8)
            nn.ELU(inplace=True),
        )
        conv_out_dim = 64 * (IMG_H // 8) * (IMG_W // 8)   # 64*8*8 = 4096
        self.proj = nn.Sequential(
            nn.Linear(conv_out_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ELU(inplace=True),
        )
        if aux_dim > 0:
            self.aux_head = nn.Linear(out_dim, aux_dim)

    def forward(self, depth: torch.Tensor):
        """
        Parameters
        ----------
        depth : (B, 1, H, W) float32, normalized to [0, 1]

        Returns
        -------
        feat : (B, out_dim)
        aux  : (B, aux_dim), returned as a tuple only if aux_dim > 0
        """
        x = self.conv(depth)                  # (B, 64, 8, 8)
        x = x.flatten(start_dim=1)            # (B, 4096)
        feat = self.proj(x)                   # (B, out_dim)
        if self.aux_dim > 0:
            return feat, self.aux_head(feat)  # (B, out_dim), (B, aux_dim)
        return feat


# -----------------------------------------------------------------------------
# Actor Network
# -----------------------------------------------------------------------------
class ActorNetwork(nn.Module):
    """
    Actor Network, CNN is fully involved from stage 1 (no alpha warm-up).

    Inputs:
        scalar_obs   : (B, SCALAR_DIM)        Scalar observations (27D), normalized
        wrist_depth  : (B, 1, IMG_H, IMG_W)   Wrist depth map, normalized to [0, 1]
        global_depth : (B, 1, IMG_H, IMG_W)   Global depth map, normalized to [0, 1]

    Outputs:
        action : (B, ACTION_DIM), squashed to (-1, 1) via tanh

    Actor does not contain hole coordinate information; it must perceive hole 
    positions via the CNN.
    """

    def __init__(self, hidden_dim: int = 256):
        super().__init__()

        # wrist_encoder aux: predicts unit_to_target (3D)
        # global_encoder aux: predicts hole_center (3D), global view is better for hole perception
        self.wrist_encoder = DepthEncoder(VISUAL_FEAT_DIM, aux_dim=3)
        self.global_encoder = DepthEncoder(VISUAL_FEAT_DIM, aux_dim=3)

        # -- Scalar + Visual Feature Fusion MLP, outputs Gaussian mean --
        # Structure: 91 → 256·ELU → 128·ELU → 5·tanh (squash mean to (-1, 1))
        self.mlp = nn.Sequential(
            nn.Linear(ACTOR_INPUT_DIM, hidden_dim),   # 91 → 256
            nn.ELU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim // 2),   # 256 → 128
            nn.ELU(inplace=True),
            nn.Linear(hidden_dim // 2, ACTION_DIM),   # 128 → 5
            nn.Tanh(),                                # Squash mean to (-1, 1)
        )

        # -- Learnable log_std, independent of input (state-independent std) --
        self.log_std = nn.Parameter(torch.zeros(ACTION_DIM))

        self._init_weights()

    def _init_weights(self):
        """Orthogonal initialization for linear layers. Output layer uses smaller gain to prevent early saturation."""
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.zeros_(m.bias)
        last_linear = [m for m in self.mlp if isinstance(m, nn.Linear)][-1]
        nn.init.orthogonal_(last_linear.weight, gain=0.01)

    def encode(
        self,
        scalar_obs:   torch.Tensor,
        wrist_depth:  torch.Tensor,
        global_depth: torch.Tensor,
        stage:        int = 1,
    ) -> torch.Tensor:
        """
        Calculates the actor_fused vector (91 dimensions).

        Returns
        -------
        fused : (B, 91)
        """
        wrist_feat, _ = self.wrist_encoder(
            wrist_depth)    # (B, 32), discard aux
        global_feat, _ = self.global_encoder(
            global_depth)  # (B, 32), discard aux
        return torch.cat([scalar_obs, wrist_feat, global_feat], dim=1)

    def forward(
        self,
        scalar_obs:   torch.Tensor,
        wrist_depth:  torch.Tensor,
        global_depth: torch.Tensor,
        stage:        int = 1,
    ) -> torch.Tensor:
        """
        Returns
        -------
        mean : (B, ACTION_DIM), in range (-1, 1), the mean of the Gaussian distribution
        """
        fused = self.encode(scalar_obs, wrist_depth,
                            global_depth, stage)  # (B, 91)
        # (B, 5)
        return self.mlp(fused)

    def get_dist(
        self,
        scalar_obs:   torch.Tensor,
        wrist_depth:  torch.Tensor,
        global_depth: torch.Tensor,
        stage:        int = 1,
    ) -> Normal:
        """
        Returns the Gaussian distribution Normal(mean, std) for the current policy.

        Returns
        -------
        dist : Normal, shape (B, ACTION_DIM)
        """
        mean = self.forward(scalar_obs, wrist_depth,
                            global_depth, stage)  # (B, 5)
        std = torch.clamp(self.log_std, -20,
                          2).exp().expand_as(mean)     # (B, 5)
        return Normal(mean, std)

    def get_dist_with_aux(
        self,
        scalar_obs:   torch.Tensor,
        wrist_depth:  torch.Tensor,
        global_depth: torch.Tensor,
        stage:        int = 1,  # noqa: ARG002  Kept for signature consistency with get_dist
    ):
        """
        Returns the policy distribution and auxiliary predictions (unit_to_target).
        Used in ppo.py update() to calculate aux_loss without rerunning the encoder.

        Returns
        -------
        dist        : Normal  shape (B, ACTION_DIM)
        target_pred : Tensor  shape (B, 3), predicted unit_to_target (normalized space)
        hole_pred   : Tensor  shape (B, 3), predicted hole_center (normalized space)
        """
        wrist_feat,  target_pred = self.wrist_encoder(
            wrist_depth)    # (B,32), (B,3)
        global_feat, hole_pred = self.global_encoder(
            global_depth)  # (B,32), (B,3)
        fused = torch.cat([scalar_obs, wrist_feat, global_feat], dim=1)
        mean = self.mlp(fused)
        std = torch.clamp(self.log_std, -20, 2).exp().expand_as(mean)
        return Normal(mean, std), target_pred, hole_pred

# -----------------------------------------------------------------------------
# Critic Network (Used during training, discarded during deployment)
# -----------------------------------------------------------------------------


class CriticNetwork(nn.Module):
    """
    Asymmetric Critic receiving pure scalar full-state observations.
    Fully decoupled from the Actor.

    Inputs:
        critic_obs : (B, 44)   Critic-specific full state observation (scalar only, no CNN)

    Outputs:
        value : (B, 1)   State value estimate V(s)

    MLP Structure: 44 → 256·ELU → 256·ELU → 128·ELU → 1·linear
    """

    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        # Structure: 44 → 256·ELU → 256·ELU → 128·ELU → 1
        self.mlp = nn.Sequential(
            nn.Linear(CRITIC_INPUT_DIM, hidden_dim),       # 44 → 256
            nn.ELU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),              # 256 → 256
            nn.ELU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim // 2),         # 256 → 128
            nn.ELU(inplace=True),
            nn.Linear(hidden_dim // 2, 1),                  # 128 → 1
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.zeros_(m.bias)
        last_linear = [m for m in self.mlp if isinstance(m, nn.Linear)][-1]
        nn.init.orthogonal_(last_linear.weight, gain=0.01)

    def forward(self, critic_obs: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        critic_obs : (B, 44)  Critic full state observation (scalar, independent of actor.encode())

        Returns
        -------
        value : (B, 1)
        """
        return self.mlp(critic_obs)   # (B, 1)
