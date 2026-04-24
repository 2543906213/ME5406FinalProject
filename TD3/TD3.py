import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import numpy as np

from network_TD3 import TD3_Actor, TD3_Critic


class TD3(object):
    def __init__(self, args):
        self.device = torch.device("cpu")
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_action = args.max_action
        self.gamma = args.gamma
        self.tau = args.tau

        # Core TD3 Parameters
        self.policy_noise = 0.2     # Target policy smoothing noise
        self.noise_clip = 0.5       # Range to clip the noise
        # Delayed policy updates: Actor updates less frequently than Critic
        self.policy_freq = 2

        self.total_it = 0           # Keep track of the number of updates

        # 1. Initialize Actor Network (passing only max_action)
        self.actor = TD3_Actor(max_action=args.max_action).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=args.lr_a)

        # 2. Initialize Twin Critic Networks
        self.critic = TD3_Critic().to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=args.lr_c)

    def select_action(self, scalar, wrist, global_d):
        """Used for interaction: Convert input to Tensors and output the action"""
        with torch.no_grad():
            s = torch.tensor(scalar, dtype=torch.float).unsqueeze(
                0).to(self.device)
            w = torch.tensor(wrist, dtype=torch.float).unsqueeze(
                0).unsqueeze(0).to(self.device)
            g = torch.tensor(global_d, dtype=torch.float).unsqueeze(
                0).unsqueeze(0).to(self.device)

            action = self.actor(s, w, g)
        return action.cpu().data.numpy().flatten()

    def update(self, replay_buffer, batch_size=256):
        """Core TD3 update logic"""
        self.total_it += 1

        # 1. Random sampling from Buffer
        s, priv, wrist_d, global_d, a, r, s_, priv_, wrist_d_, global_d_, dw = replay_buffer.sample(
            batch_size)

        with torch.no_grad():
            # 2. Target Policy Smoothing: Add noise to the action of the next state to improve robustness
            noise = (torch.randn_like(a) *
                     self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            # noise = (torch.randn(a.shape, device="cpu").to(self.device) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)

            # Target Actor outputs the action and adds noise
            next_action = (self.actor_target(s_, wrist_d_, global_d_) +
                           noise).clamp(-self.max_action, self.max_action)

            # 3. Calculate Target Q-value: Use the minimum of Twin Critics to mitigate overestimation
            target_Q1, target_Q2 = self.critic_target(priv_, next_action)
            target_Q = r + self.gamma * \
                (1 - dw) * torch.min(target_Q1, target_Q2)

        # 4. Update the current Critic Networks
        current_Q1, current_Q2 = self.critic(priv, a)
        critic_loss = F.mse_loss(current_Q1, target_Q) + \
            F.mse_loss(current_Q2, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # 5. Delayed Policy Update: Update Actor and Target networks every policy_freq iterations
        metrics = {"critic_loss": critic_loss.item(), "actor_loss": 0.0}

        if self.total_it % self.policy_freq == 0:
            # Calculate Actor Loss: Maximize the Q1 value
            # Note: During Actor update, we need to recalculate the action through the current Actor
            actor_loss = - \
                self.critic.Q1(priv, self.actor(s, wrist_d, global_d)).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # 6. Soft Update target networks
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data)

            metrics["actor_loss"] = actor_loss.item()

        return metrics

    def save(self, path, stage=1, obs_norm=None, priv_norm=None):
        """Save weights and normalization state"""
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "stage": stage,
            "obs_norm": obs_norm,
            "priv_norm": priv_norm,
        }, path)

    def load(self, path):
        """Load weights"""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.actor_target = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)
        return ckpt.get("stage", 1)
