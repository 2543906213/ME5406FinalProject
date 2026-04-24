import os
import numpy as np
import torch
import torch.nn.functional as F

from SAC.network import SACActor, SACQNetwork

# The position of unit_to_target in scalar obs is consistent with PPO
UNIT_TO_TARGET_IDX = slice(25, 28)


class SACAgent:
    def __init__(self, args):
        self.hidden_dim = args.hidden_dim
        self.gamma = args.gamma
        self.tau = args.tau
        self.batch_size = args.batch_size
        self.updates_per_step = args.updates_per_step

        self.lr_actor = args.lr_actor
        self.lr_critic = args.lr_critic
        self.lr_alpha = args.lr_alpha
        self.init_alpha = float(args.init_alpha)

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        print(f"[SAC] use device: {self.device}")

        self.actor = SACActor(hidden_dim=self.hidden_dim).to(self.device)
        self.critic1 = SACQNetwork(hidden_dim=self.hidden_dim).to(self.device)
        self.critic2 = SACQNetwork(hidden_dim=self.hidden_dim).to(self.device)
        self.target_critic1 = SACQNetwork(
            hidden_dim=self.hidden_dim).to(self.device)
        self.target_critic2 = SACQNetwork(
            hidden_dim=self.hidden_dim).to(self.device)

        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

        self.optimizer_actor = torch.optim.Adam(
            self.actor.parameters(), lr=self.lr_actor)
        self.optimizer_critic1 = torch.optim.Adam(
            self.critic1.parameters(), lr=self.lr_critic)
        self.optimizer_critic2 = torch.optim.Adam(
            self.critic2.parameters(), lr=self.lr_critic)

        # automatic entropy temperature alpha
        self.log_alpha = torch.tensor(
            np.log(self.init_alpha), dtype=torch.float32, device=self.device)
        self.log_alpha.requires_grad = True
        self.optimizer_alpha = torch.optim.Adam(
            [self.log_alpha], lr=self.lr_alpha)

        self.target_entropy = -float(args.action_dim)
        self.aux_coef = getattr(args, "aux_coef", 0.3)

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def select_action(self, scalar_obs, wrist_depth, global_depth, deterministic=False):
        s = torch.tensor(scalar_obs, dtype=torch.float32,
                         device=self.device).unsqueeze(0)
        wd = torch.tensor(wrist_depth, dtype=torch.float32,
                          device=self.device).unsqueeze(0).unsqueeze(0)
        gd = torch.tensor(global_depth, dtype=torch.float32,
                          device=self.device).unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            action, _ = self.actor.sample(
                s, wd, gd, deterministic=deterministic, with_logprob=False)
        return action.cpu().numpy().flatten()

    def _soft_update(self, src_net, dst_net):
        for src_p, dst_p in zip(src_net.parameters(), dst_net.parameters()):
            dst_p.data.mul_(1.0 - self.tau)
            dst_p.data.add_(self.tau * src_p.data)

    def update(self, replay_buffer):
        if replay_buffer.size < self.batch_size:
            return None

        actor_loss_list = []
        critic1_loss_list = []
        critic2_loss_list = []
        alpha_loss_list = []
        alpha_value_list = []
        aux_loss_list = []

        for _ in range(self.updates_per_step):
            batch = replay_buffer.sample(self.batch_size, self.device)

            s = batch["s"]
            priv = batch["priv"]
            wd = batch["wrist"]
            gd = batch["global_d"]
            a = batch["a"]
            r = batch["r"]

            s_next = batch["s_next"]
            priv_next = batch["priv_next"]
            wd_next = batch["wrist_next"]
            gd_next = batch["global_d_next"]
            dw = batch["dw"]

            # 1) Update double Q
            with torch.no_grad():
                next_a, next_logp = self.actor.sample(
                    s_next, wd_next, gd_next, deterministic=False, with_logprob=True)
                target_q1 = self.target_critic1(priv_next, next_a)
                target_q2 = self.target_critic2(priv_next, next_a)
                target_q = torch.min(target_q1, target_q2) - \
                    self.alpha.detach() * next_logp
                target = r + self.gamma * (1.0 - dw) * target_q

            q1 = self.critic1(priv, a)
            q2 = self.critic2(priv, a)

            critic1_loss = F.mse_loss(q1, target)
            critic2_loss = F.mse_loss(q2, target)

            self.optimizer_critic1.zero_grad()
            critic1_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic1.parameters(), 0.5)
            self.optimizer_critic1.step()

            self.optimizer_critic2.zero_grad()
            critic2_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), 0.5)
            self.optimizer_critic2.step()

            # 2) Update policy (with CNN aux_loss)
            pi_a, logp_pi, target_pred, hole_pred = self.actor.sample_with_aux(
                s, wd, gd)
            q1_pi = self.critic1(priv, pi_a)
            q2_pi = self.critic2(priv, pi_a)
            min_q_pi = torch.min(q1_pi, q2_pi)

            actor_loss = (self.alpha.detach() * logp_pi - min_q_pi).mean()

            unit_gt = s[:, UNIT_TO_TARGET_IDX].detach()
            aux_loss = F.mse_loss(target_pred, unit_gt) + \
                F.mse_loss(hole_pred, unit_gt)

            self.optimizer_actor.zero_grad()
            (actor_loss + self.aux_coef * aux_loss).backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
            self.optimizer_actor.step()

            # 3) Update temperature alpha (skip during freezing period)
            if self.log_alpha.requires_grad:
                alpha_loss = -(self.log_alpha * (logp_pi +
                               self.target_entropy).detach()).mean()
                self.optimizer_alpha.zero_grad()
                alpha_loss.backward()
                self.optimizer_alpha.step()
            else:
                alpha_loss = torch.tensor(0.0)

            # 4) Soft update target Q
            self._soft_update(self.critic1, self.target_critic1)
            self._soft_update(self.critic2, self.target_critic2)

            actor_loss_list.append(float(actor_loss.item()))
            critic1_loss_list.append(float(critic1_loss.item()))
            critic2_loss_list.append(float(critic2_loss.item()))
            alpha_loss_list.append(float(alpha_loss.item()))
            alpha_value_list.append(float(self.alpha.item()))
            aux_loss_list.append(float(aux_loss.item()))

        # CNN Healthiness (feature statistics of the last mini-batch)
        with torch.no_grad():
            wrist_feat,  _ = self.actor.wrist_encoder(wd[:64])
            global_feat, _ = self.actor.global_encoder(gd[:64])
        wrist_gnorm = global_gnorm = 0.0
        for p in self.actor.wrist_encoder.parameters():
            if p.grad is not None:
                wrist_gnorm += p.grad.norm().item() ** 2
        for p in self.actor.global_encoder.parameters():
            if p.grad is not None:
                global_gnorm += p.grad.norm().item() ** 2

        return {
            "actor_loss":      float(np.mean(actor_loss_list)),
            "critic1_loss":    float(np.mean(critic1_loss_list)),
            "critic2_loss":    float(np.mean(critic2_loss_list)),
            "alpha_loss":      float(np.mean(alpha_loss_list)),
            "alpha":           float(np.mean(alpha_value_list)),
            "aux_loss":        float(np.mean(aux_loss_list)),
            "wrist_feat_std":  float(wrist_feat.std().item()),
            "global_feat_std": float(global_feat.std().item()),
            "wrist_gnorm":     wrist_gnorm ** 0.5,
            "global_gnorm":    global_gnorm ** 0.5,
        }

    def save(self, path: str, stage: int = 0, obs_norm=None, priv_norm=None):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        data = {
            "actor": self.actor.state_dict(),
            "critic1": self.critic1.state_dict(),
            "critic2": self.critic2.state_dict(),
            "target_critic1": self.target_critic1.state_dict(),
            "target_critic2": self.target_critic2.state_dict(),
            "optimizer_actor": self.optimizer_actor.state_dict(),
            "optimizer_critic1": self.optimizer_critic1.state_dict(),
            "optimizer_critic2": self.optimizer_critic2.state_dict(),
            "log_alpha": float(self.log_alpha.detach().cpu().item()),
            "optimizer_alpha": self.optimizer_alpha.state_dict(),
            "hidden_dim": self.hidden_dim,
            "stage": stage,
        }

        if obs_norm is not None:
            data["obs_norm_n"] = obs_norm.running_ms.n
            data["obs_norm_mean"] = obs_norm.running_ms.mean.copy()
            data["obs_norm_S"] = obs_norm.running_ms.S.copy()
            data["obs_norm_std"] = obs_norm.running_ms.std.copy()

        if priv_norm is not None:
            data["priv_norm_n"] = priv_norm.running_ms.n
            data["priv_norm_mean"] = priv_norm.running_ms.mean.copy()
            data["priv_norm_S"] = priv_norm.running_ms.S.copy()
            data["priv_norm_std"] = priv_norm.running_ms.std.copy()

        torch.save(data, path)

    def load(self, path: str, reset_optimizer: bool = False) -> int:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic1.load_state_dict(ckpt["critic1"])
        self.critic2.load_state_dict(ckpt["critic2"])
        self.target_critic1.load_state_dict(ckpt["target_critic1"])
        self.target_critic2.load_state_dict(ckpt["target_critic2"])

        if not reset_optimizer:
            self.optimizer_actor.load_state_dict(ckpt["optimizer_actor"])
            self.optimizer_critic1.load_state_dict(ckpt["optimizer_critic1"])
            self.optimizer_critic2.load_state_dict(ckpt["optimizer_critic2"])
            self.log_alpha = torch.tensor(
                float(ckpt["log_alpha"]), dtype=torch.float32, device=self.device)
            self.log_alpha.requires_grad = True
            self.optimizer_alpha = torch.optim.Adam(
                [self.log_alpha], lr=self.lr_alpha)
            self.optimizer_alpha.load_state_dict(ckpt["optimizer_alpha"])
        else:
            self.optimizer_actor.state.clear()
            self.optimizer_critic1.state.clear()
            self.optimizer_critic2.state.clear()
            self.log_alpha = torch.tensor(
                np.log(self.init_alpha), dtype=torch.float32, device=self.device)
            self.log_alpha.requires_grad = True
            self.optimizer_alpha = torch.optim.Adam(
                [self.log_alpha], lr=self.lr_alpha)

        return int(ckpt.get("stage", 0))

    def reset_lr(self, lr_parameter: float = 0.5):
        """Called when switching course stages: raise the lr part and clear the Adam momentum cache"""
        for p in self.optimizer_actor.param_groups:
            p["lr"] = (1 - lr_parameter) * self.lr_actor + \
                lr_parameter * p["lr"]
        for p in self.optimizer_critic1.param_groups:
            p["lr"] = (1 - lr_parameter) * self.lr_critic + \
                lr_parameter * p["lr"]
        for p in self.optimizer_critic2.param_groups:
            p["lr"] = (1 - lr_parameter) * self.lr_critic + \
                lr_parameter * p["lr"]
        self.optimizer_actor.state.clear()
        self.optimizer_critic1.state.clear()
        self.optimizer_critic2.state.clear()
