import torch
import torch.nn as nn
from torch.distributions import Normal

from common.env import (
    SCALAR_DIM,
    PRIVILEGED_DIM,
    ACTION_DIM,
    IMG_H,
    IMG_W,
)

VISUAL_FEAT_DIM = 32
ACTOR_INPUT_DIM = SCALAR_DIM + VISUAL_FEAT_DIM * 2
Q_INPUT_DIM = PRIVILEGED_DIM + ACTION_DIM
LOG_STD_MIN = -20
LOG_STD_MAX = 2


class DepthEncoder(nn.Module):
    """Encodes single-channel depth map (B,1,64,64) into a low-dim feature vector, with optional aux head."""

    def __init__(self, out_dim: int = VISUAL_FEAT_DIM, aux_dim: int = 0):
        super().__init__()
        self.aux_dim = aux_dim
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
            nn.LayerNorm(out_dim),
            nn.ELU(inplace=True),
        )
        if aux_dim > 0:
            self.aux_head = nn.Linear(out_dim, aux_dim)

    def forward(self, depth: torch.Tensor):
        x = self.conv(depth)
        x = x.flatten(start_dim=1)
        feat = self.proj(x)
        if self.aux_dim > 0:
            return feat, self.aux_head(feat)
        return feat


class SACActor(nn.Module):
    """SAC Stochastic Policy Network: takes scalar + dual depth maps, outputs tanh-squashed Gaussian policy."""

    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        self.wrist_encoder = DepthEncoder(VISUAL_FEAT_DIM, aux_dim=3)
        self.global_encoder = DepthEncoder(VISUAL_FEAT_DIM, aux_dim=3)

        self.backbone = nn.Sequential(
            nn.Linear(ACTOR_INPUT_DIM, hidden_dim),
            nn.ELU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(inplace=True),
        )
        self.mean_head = nn.Linear(hidden_dim, ACTION_DIM)
        self.log_std_head = nn.Linear(hidden_dim, ACTION_DIM)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.mean_head.weight, gain=0.01)
        nn.init.orthogonal_(self.log_std_head.weight, gain=0.01)

    def encode(
        self,
        scalar_obs: torch.Tensor,
        wrist_depth: torch.Tensor,
        global_depth: torch.Tensor,
    ) -> torch.Tensor:
        wrist_feat,  _ = self.wrist_encoder(wrist_depth)
        global_feat, _ = self.global_encoder(global_depth)
        return torch.cat([scalar_obs, wrist_feat, global_feat], dim=1)

    def sample_with_aux(
        self,
        scalar_obs: torch.Tensor,
        wrist_depth: torch.Tensor,
        global_depth: torch.Tensor,
    ):
        """Similar to sample(), but returns aux predictions from two CNNs for aux_loss calculation."""
        wrist_feat,  target_pred = self.wrist_encoder(wrist_depth)
        global_feat, hole_pred = self.global_encoder(global_depth)
        fused = torch.cat([scalar_obs, wrist_feat, global_feat], dim=1)
        h = self.backbone(fused)
        mean = self.mean_head(h)
        log_std = torch.clamp(self.log_std_head(h), LOG_STD_MIN, LOG_STD_MAX)
        dist = Normal(mean, log_std.exp())
        z = dist.rsample()
        action = torch.tanh(z)
        log_prob = (dist.log_prob(z) - torch.log(1.0 -
                    action.pow(2) + 1e-6)).sum(1, keepdim=True)
        return action, log_prob, target_pred, hole_pred

    def forward(
        self,
        scalar_obs: torch.Tensor,
        wrist_depth: torch.Tensor,
        global_depth: torch.Tensor,
    ):
        fused = self.encode(scalar_obs, wrist_depth, global_depth)
        h = self.backbone(fused)
        mean = self.mean_head(h)
        log_std = self.log_std_head(h)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(
        self,
        scalar_obs: torch.Tensor,
        wrist_depth: torch.Tensor,
        global_depth: torch.Tensor,
        deterministic: bool = False,
        with_logprob: bool = True,
    ):
        mean, log_std = self.forward(scalar_obs, wrist_depth, global_depth)
        std = log_std.exp()
        dist = Normal(mean, std)

        if deterministic:
            z = mean
        else:
            z = dist.rsample()

        action = torch.tanh(z)

        if not with_logprob:
            return action, None

        # Correction term for tanh transformation to ensure correct log_prob
        log_prob = dist.log_prob(z) - torch.log(1.0 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=1, keepdim=True)
        return action, log_prob


class SACQNetwork(nn.Module):
    """Q(s,a) Network: Asymmetric input using privileged_obs + action."""

    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(Q_INPUT_DIM, hidden_dim),
            nn.ELU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ELU(inplace=True),
            nn.Linear(hidden_dim // 2, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.zeros_(m.bias)
        last_linear = [m for m in self.mlp if isinstance(m, nn.Linear)][-1]
        nn.init.orthogonal_(last_linear.weight, gain=0.01)

    def forward(self, priv_obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([priv_obs, action], dim=1)
        return self.mlp(x)
