import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from PPO.network import ActorNetwork, CriticNetwork
from common.normalization import Normalization


class PPO_continuous():
    def __init__(self, args):
        self.max_action = args.max_action
        self.batch_size = args.batch_size
        self.mini_batch_size = args.mini_batch_size
        self.max_train_steps = args.max_train_steps
        self.lr_a = args.lr_a  # Learning rate of actor
        self.lr_c = args.lr_c  # Learning rate of critic
        self.gamma = args.gamma  # Discount factor
        self.lamda = args.lamda  # GAE parameter
        self.epsilon = args.epsilon  # PPO clip parameter
        self.K_epochs = args.K_epochs  # PPO parameter
        # Entropy coefficient (temperature coefficient)
        self.entropy_coef = args.entropy_coef
        self.set_adam_eps = args.set_adam_eps
        self.use_grad_clip = args.use_grad_clip
        self.use_lr_decay = args.use_lr_decay  # Learning rate decay parameter
        self.lr_parameter = args.lr_parameter  # Learning rate recovery parameter
        self.use_adv_norm = args.use_adv_norm  # Advantage normalization parameter
        _kl = getattr(args, "target_kl", None)
        # Disable if ≤0 or not set
        self.target_kl = _kl if (_kl is not None and _kl > 0) else None
        self.aux_coef = getattr(args, "aux_coef", 0.3)  # Weight for aux_loss

        # Trick 1: Advantage normalizer, updates stats online
        self.adv_norm = Normalization(shape=1)

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        print(f"[PPO] Using device: {self.device}")

        self.actor = ActorNetwork().to(self.device)
        self.critic = CriticNetwork().to(self.device)

        if self.set_adam_eps:  # Trick 9: set Adam epsilon=1e-5
            self.optimizer_actor = torch.optim.Adam(
                self.actor.parameters(), lr=self.lr_a, eps=1e-5)
            self.optimizer_critic = torch.optim.Adam(
                self.critic.parameters(), lr=self.lr_c, eps=1e-5)
        else:
            self.optimizer_actor = torch.optim.Adam(
                self.actor.parameters(), lr=self.lr_a)
            self.optimizer_critic = torch.optim.Adam(
                self.critic.parameters(), lr=self.lr_c)

    def interact(self, scalar_obs, wrist_depth, global_depth, stage=1):
        """
        Interact with the environment: sample an action based on current observations, 
        returning the action and corresponding log_prob. 
        The train loop calls buffer.store() after obtaining r / s_ / dw / done.

        Parameters
        ----------
        scalar_obs   : np.ndarray (SCALAR_DIM,)    Normalized scalar observations
        wrist_depth  : np.ndarray (H, W)            Depth map normalized to [0,1]
        global_depth : np.ndarray (H, W)            Depth map normalized to [0,1]
        stage        : int 1/2/3/4                  Controls CNN alpha weights

        Returns
        -------
        action    : np.ndarray (ACTION_DIM,)  Clipped to [-max_action, max_action]
        a_logprob : np.ndarray (ACTION_DIM,)  Corresponding log π(a|s)
        """
        # numpy → tensor, add batch and channel dimensions
        s = torch.tensor(scalar_obs,   dtype=torch.float).unsqueeze(
            0).to(self.device)          # (1, 27)
        wd = torch.tensor(wrist_depth,  dtype=torch.float).unsqueeze(
            0).unsqueeze(0).to(self.device)  # (1,1,H,W)
        gd = torch.tensor(global_depth, dtype=torch.float).unsqueeze(
            0).unsqueeze(0).to(self.device)  # (1,1,H,W)

        with torch.no_grad():   # No gradient needed for sampling
            dist = self.actor.get_dist(s, wd, gd, stage)  # Normal(mean, std)
            a = dist.sample()                           # (1, ACTION_DIM)
            a = torch.clamp(a, -self.max_action, self.max_action)
            # (1, ACTION_DIM)
            a_logprob = dist.log_prob(a)

        return a.cpu().numpy().flatten(), a_logprob.cpu().numpy().flatten()

    def evaluate(self, scalar_obs, wrist_depth, global_depth, stage=1):
        """Use policy mean as action during evaluation (no sampling, no computation graph)."""
        s = torch.tensor(scalar_obs,   dtype=torch.float).unsqueeze(
            0).to(self.device)
        wd = torch.tensor(wrist_depth,  dtype=torch.float).unsqueeze(
            0).unsqueeze(0).to(self.device)
        gd = torch.tensor(global_depth, dtype=torch.float).unsqueeze(
            0).unsqueeze(0).to(self.device)
        with torch.no_grad():
            # forward() directly returns tanh(mean)
            action = self.actor(s, wd, gd, stage)
        return action.cpu().numpy().flatten()

    def update(self, replay_buffer, total_steps, stage):
        (s, priv, wrist_d, global_d,
         a, a_logprob, r,
         s_, priv_, wrist_d_, global_d_,
         dw, done) = [t.to(self.device) for t in replay_buffer.numpy_to_tensor()]

        # -- GAE Calculation ----------------------------------------------------------
        adv = []
        gae = 0
        with torch.no_grad():
            # Critic independently receives pure scalar critic_obs (priv), no actor.encode() dependency
            vs = self.critic(priv)    # (B, 1)
            vs_ = self.critic(priv_)

            deltas = r + self.gamma * (1.0 - dw) * vs_ - vs
            for delta, d in zip(reversed(deltas.cpu().flatten().numpy()),
                                reversed(done.cpu().flatten().numpy())):
                gae = delta + self.gamma * self.lamda * gae * (1.0 - d)
                adv.insert(0, gae)

            adv = torch.tensor(
                adv, dtype=torch.float).view(-1, 1).to(self.device)
            v_target = adv + vs

            if self.use_adv_norm:  # Trick 1: advantage normalization
                adv_np = np.array([float(self.adv_norm(v.item()))
                                  for v in adv.cpu().flatten()])
                adv = torch.tensor(
                    adv_np, dtype=torch.float).view(-1, 1).to(self.device)

        # -- K mini-batch Update Epochs ----------------------------------------------
        actor_losses, critic_losses, aux_losses = [], [], []
        kl_early_stopped = False

        # unit_to_target is located at [25:28] in normalized scalar observations
        # (6+6+6+3+4=25, followed by 3 dims)
        UNIT_TO_TARGET_IDX = slice(25, 28)

        for _ in range(self.K_epochs):
            for index in BatchSampler(
                SubsetRandomSampler(range(self.batch_size)
                                    ), self.mini_batch_size, False
            ):
                # Actor update (simultaneously get auxiliary predictions for both CNNs)
                dist_now, target_pred, hole_pred = self.actor.get_dist_with_aux(
                    s[index], wrist_d[index], global_d[index], stage)
                dist_entropy = dist_now.entropy().sum(1, keepdim=True)   # (mini_B, 1)
                a_logprob_now = dist_now.log_prob(a[index])
                ratios = torch.exp(
                    a_logprob_now.sum(1, keepdim=True) -
                    a_logprob[index].sum(1, keepdim=True)
                )   # (mini_B, 1)

                # KL Early Stopping: approx_kl = mean(log π_old - log π_new)
                if self.target_kl is not None:
                    with torch.no_grad():
                        approx_kl = (a_logprob[index].sum(
                            1) - a_logprob_now.sum(1)).mean().item()
                    if approx_kl > self.target_kl:
                        kl_early_stopped = True
                        break

                surr1 = ratios * adv[index]
                surr2 = torch.clamp(ratios, 1 - self.epsilon,
                                    1 + self.epsilon) * adv[index]
                actor_loss = -torch.min(surr1, surr2) - \
                    self.entropy_coef * dist_entropy

                # wrist aux: predict unit_to_target (scalar obs [25:28], valid before/after insertion)
                unit_to_target_gt = s[index][:, UNIT_TO_TARGET_IDX].detach()
                wrist_aux_loss = F.mse_loss(target_pred, unit_to_target_gt)
                # global aux: also predict unit_to_target (global camera is behind the board, better ball view)
                global_aux_loss = F.mse_loss(hole_pred,   unit_to_target_gt)
                aux_loss = wrist_aux_loss + global_aux_loss

                self.optimizer_actor.zero_grad()
                (actor_loss.mean() + self.aux_coef * aux_loss).backward()
                if self.use_grad_clip:
                    torch.nn.utils.clip_grad_norm_(
                        self.actor.parameters(), 0.5)
                self.optimizer_actor.step()
                actor_losses.append(actor_loss.mean().item())
                aux_losses.append(aux_loss.item())

                # Critic update (decoupled from Actor, receives pure scalar critic_obs)
                v_s = self.critic(priv[index])
                critic_loss = F.mse_loss(v_target[index], v_s)

                self.optimizer_critic.zero_grad()
                critic_loss.backward()
                if self.use_grad_clip:
                    torch.nn.utils.clip_grad_norm_(
                        self.critic.parameters(), 0.5)
                self.optimizer_critic.step()
                critic_losses.append(critic_loss.item())

            if kl_early_stopped:
                break

        if self.use_lr_decay:  # Trick 6: Learning rate Decay
            self.lr_decay(total_steps)

        # -- CNN Health Metrics -------------------------------------------------------
        with torch.no_grad():
            wrist_feat,  _ = self.actor.wrist_encoder(wrist_d[:64])
            global_feat, _ = self.actor.global_encoder(global_d[:64])
            wrist_feat_std = float(wrist_feat.std().item())
            global_feat_std = float(global_feat.std().item())

        wrist_grad_norm = global_grad_norm = 0.0
        for p in self.actor.wrist_encoder.parameters():
            if p.grad is not None:
                wrist_grad_norm += p.grad.norm().item() ** 2
        for p in self.actor.global_encoder.parameters():
            if p.grad is not None:
                global_grad_norm += p.grad.norm().item() ** 2
        wrist_grad_norm = wrist_grad_norm ** 0.5
        global_grad_norm = global_grad_norm ** 0.5

        return {
            "actor_loss":       sum(actor_losses) / len(actor_losses),
            "critic_loss":      sum(critic_losses) / len(critic_losses),
            "aux_loss":         sum(aux_losses) / len(aux_losses),
            "wrist_feat_std":   wrist_feat_std,
            "global_feat_std":  global_feat_std,
            "wrist_grad_norm":  wrist_grad_norm,
            "global_grad_norm": global_grad_norm,
        }

    def lr_decay(self, total_steps):
        """Learning rate decay: as training steps increase, LR decreases to aid stable convergence."""
        lr_a_now = self.lr_a * (1 - total_steps / self.max_train_steps)
        lr_c_now = self.lr_c * (1 - total_steps / self.max_train_steps)
        for p in self.optimizer_actor.param_groups:
            p['lr'] = lr_a_now
        for p in self.optimizer_critic.param_groups:
            p['lr'] = lr_c_now

    def reset_lr(self):
        """Called when switching curriculum stages: increases LR and clears Adam momentum buffers."""
        for p in self.optimizer_actor.param_groups:
            p['lr'] = (1 - self.lr_parameter) * self.lr_a + \
                self.lr_parameter * (p['lr'])
        for p in self.optimizer_critic.param_groups:
            p['lr'] = (1 - self.lr_parameter) * self.lr_c + \
                self.lr_parameter * (p['lr'])
        self.optimizer_actor.state.clear()
        self.optimizer_critic.state.clear()

    def save(self, path: str, stage: int = 0, obs_norm=None, priv_norm=None):
        """
        Save actor/critic weights + optimizer states + adv_norm statistics.
        obs_norm / priv_norm are env.scalar_norm / env.priv_norm (Normalization objects).
        If provided, they are saved together to ensure normalization consistency during eval.
        """
        import os
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        d = {
            "actor":            self.actor.state_dict(),
            "critic":           self.critic.state_dict(),
            "optimizer_actor":  self.optimizer_actor.state_dict(),
            "optimizer_critic": self.optimizer_critic.state_dict(),
            "adv_norm_n":    self.adv_norm.running_ms.n,
            "adv_norm_mean": self.adv_norm.running_ms.mean,
            "adv_norm_S":    self.adv_norm.running_ms.S,
            "adv_norm_std":  self.adv_norm.running_ms.std,
            "stage":         stage,
        }
        if obs_norm is not None:
            d["obs_norm_n"] = obs_norm.running_ms.n
            d["obs_norm_mean"] = obs_norm.running_ms.mean.copy()
            d["obs_norm_S"] = obs_norm.running_ms.S.copy()
            d["obs_norm_std"] = obs_norm.running_ms.std.copy()
        if priv_norm is not None:
            d["priv_norm_n"] = priv_norm.running_ms.n
            d["priv_norm_mean"] = priv_norm.running_ms.mean.copy()
            d["priv_norm_S"] = priv_norm.running_ms.S.copy()
            d["priv_norm_std"] = priv_norm.running_ms.std.copy()
        torch.save(d, path)

    def load(self, path: str, reset_optimizer: bool = False) -> int:
        """
        Recover weights from checkpoint and return the stage (returns 0 if not recorded).

        Parameters
        ----------
        reset_optimizer : bool
            True  — Only load network weights, do not restore optimizer states or adv_norm stats.
                    Useful for fine-tuning / resume (prevents old Adam momentum interference).
            False — Full recovery (continue the same training run).
        """
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        missing, unexpected = self.actor.load_state_dict(
            ckpt["actor"], strict=False)
        if missing:
            print(
                f"[load] New keys in actor (randomly initialized): {missing}")
        self.critic.load_state_dict(ckpt["critic"], strict=False)
        if not reset_optimizer:
            self.optimizer_actor.load_state_dict(ckpt["optimizer_actor"])
            self.optimizer_critic.load_state_dict(ckpt["optimizer_critic"])
            self.adv_norm.running_ms.n = ckpt["adv_norm_n"]
            self.adv_norm.running_ms.mean = ckpt["adv_norm_mean"]
            self.adv_norm.running_ms.S = ckpt["adv_norm_S"]
            self.adv_norm.running_ms.std = ckpt["adv_norm_std"]
        else:
            # Clear Adam momentum, reset advantage normalizer
            self.optimizer_actor.state.clear()
            self.optimizer_critic.state.clear()
            self.adv_norm.running_ms.n = 0
            self.adv_norm.running_ms.mean = np.zeros(1)
            self.adv_norm.running_ms.S = np.zeros(1)
            self.adv_norm.running_ms.std = np.zeros(1)
        return int(ckpt.get("stage", 0))
