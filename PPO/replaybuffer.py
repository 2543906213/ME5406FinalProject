import torch
import numpy as np

# Named as rollout buffer, but technically an episode buffer. It is updated
# once after each episode ends, storing all transition data for the
# current episode to be used during PPO update().


class ReplayBuffer:
    def __init__(self, args):
        B = args.batch_size
        H = args.img_h        # Depth map height (pixels)
        W = args.img_w        # Depth map width (pixels)

        # -- Scalar Observations (Actor) + Privileged Observations (Critic) --
        self.s = np.zeros((B, args.state_dim),      dtype=np.float32)
        self.s_ = np.zeros((B, args.state_dim),      dtype=np.float32)
        self.priv = np.zeros((B, args.priv_dim),       dtype=np.float32)
        self.priv_ = np.zeros((B, args.priv_dim),       dtype=np.float32)

        # -- Depth Maps: wrist + global, current frame and next frame --
        # Shape (B, H, W), normalized to [0, 1]
        self.wrist_d = np.zeros((B, H, W), dtype=np.float32)
        self.wrist_d_ = np.zeros((B, H, W), dtype=np.float32)
        self.global_d = np.zeros((B, H, W), dtype=np.float32)
        self.global_d_ = np.zeros((B, H, W), dtype=np.float32)

        # -- Actions / Rewards / Termination Flags --
        self.a = np.zeros((B, args.action_dim),    dtype=np.float32)
        self.a_logprob = np.zeros((B, args.action_dim),   dtype=np.float32)
        self.r = np.zeros((B, 1),                  dtype=np.float32)
        self.dw = np.zeros(
            (B, 1),                  dtype=np.float32)  # dead or win
        self.done = np.zeros((B, 1),                  dtype=np.float32)

        self.count = 0

    def store(self, s, priv, wrist_d, global_d,
              a, a_logprob, r,
              s_, priv_, wrist_d_, global_d_,
              dw, done):
        """
        Stores transitions into the buffer. Called during environment interaction.
        Contains 13 parameters: current observations (scalar + privileged + depth), 
        sampled action, action log_prob, reward, next observations, 
        dead-or-win flag, and done flag.
        """
        self.s[self.count] = s
        self.priv[self.count] = priv
        self.wrist_d[self.count] = wrist_d
        self.global_d[self.count] = global_d

        self.a[self.count] = a
        self.a_logprob[self.count] = a_logprob
        self.r[self.count] = r

        self.s_[self.count] = s_
        self.priv_[self.count] = priv_
        self.wrist_d_[self.count] = wrist_d_
        self.global_d_[self.count] = global_d_

        self.dw[self.count] = dw
        self.done[self.count] = done
        self.count += 1

    def numpy_to_tensor(self):
        """
        Converts numpy episode data to tensors. Called within PPO update() 
        to feed data into the networks for calculating action distributions 
        and state values.
        """
        s = torch.tensor(self.s,         dtype=torch.float)
        priv = torch.tensor(self.priv,      dtype=torch.float)

        # Add channel dimension to depth maps -> (B, 1, H, W) to match CNN input format
        wrist_d = torch.tensor(self.wrist_d,   dtype=torch.float).unsqueeze(1)
        global_d = torch.tensor(self.global_d,  dtype=torch.float).unsqueeze(1)

        a = torch.tensor(self.a,         dtype=torch.float)
        a_logprob = torch.tensor(self.a_logprob, dtype=torch.float)
        r = torch.tensor(self.r,         dtype=torch.float)

        s_ = torch.tensor(self.s_,        dtype=torch.float)
        priv_ = torch.tensor(self.priv_,     dtype=torch.float)
        wrist_d_ = torch.tensor(self.wrist_d_,  dtype=torch.float).unsqueeze(1)
        global_d_ = torch.tensor(
            self.global_d_, dtype=torch.float).unsqueeze(1)

        dw = torch.tensor(self.dw,        dtype=torch.float)
        done = torch.tensor(self.done,      dtype=torch.float)

        return (s, priv, wrist_d, global_d,
                a, a_logprob, r,
                s_, priv_, wrist_d_, global_d_,
                dw, done)
