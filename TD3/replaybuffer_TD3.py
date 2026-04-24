import torch
import numpy as np


class ReplayBuffer:
    def __init__(self, args):
        # TD3 requires a large memory capacity, passed via args.buffer_size (e.g., 1,000,000)
        self.max_size = args.buffer_size
        self.count = 0  # Records the current write cursor position
        self.size = 0   # Records the total number of valid entries in the buffer

        H = args.img_h        # Depth map height (pixels)
        W = args.img_w        # Depth map width (pixels)

        # -- Scalar Observations (for Actor) + Privileged Observations (for Critic) ---
        self.s = np.zeros((self.max_size, args.state_dim),
                          dtype=np.float32)
        self.s_ = np.zeros((self.max_size, args.state_dim),
                           dtype=np.float32)
        self.priv = np.zeros(
            (self.max_size, args.priv_dim),       dtype=np.float32)
        self.priv_ = np.zeros(
            (self.max_size, args.priv_dim),       dtype=np.float32)

        # -- Depth Maps: wrist + global, current frame and next frame ----------------
        self.wrist_d = np.zeros((self.max_size, H, W), dtype=np.float16)
        self.wrist_d_ = np.zeros((self.max_size, H, W), dtype=np.float16)
        self.global_d = np.zeros((self.max_size, H, W), dtype=np.float16)
        self.global_d_ = np.zeros((self.max_size, H, W), dtype=np.float16)

        # -- Action / Reward / Termination flags (Removed PPO's a_logprob) ------------
        self.a = np.zeros((self.max_size, args.action_dim),
                          dtype=np.float32)
        self.r = np.zeros((self.max_size, 1),
                          dtype=np.float32)
        self.dw = np.zeros((self.max_size, 1),
                           dtype=np.float32)  # dead or win
        self.done = np.zeros((self.max_size, 1),
                             dtype=np.float32)

        # Pre-fetch device; push tensors directly to GPU during sampling to accelerate training
        # self.device = torch.device("cpu")
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

    # Store data: Note that a_logprob has been removed from the parameters
    def store(self, s, priv, wrist_d, global_d,
              a, r,
              s_, priv_, wrist_d_, global_d_,
              dw, done):

        self.s[self.count] = s
        self.priv[self.count] = priv
        self.wrist_d[self.count] = wrist_d
        self.global_d[self.count] = global_d

        self.a[self.count] = a
        self.r[self.count] = r

        self.s_[self.count] = s_
        self.priv_[self.count] = priv_
        self.wrist_d_[self.count] = wrist_d_
        self.global_d_[self.count] = global_d_

        self.dw[self.count] = dw
        self.done[self.count] = done

        # Circular buffer logic: reset count to 0 if max_size is reached, overwriting oldest data
        self.count = (self.count + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    # TD3 specific method: Random sampling (replaces PPO's numpy_to_tensor)
    def sample(self, batch_size):
        # Randomly select batch_size indices from the current buffer [0, self.size)
        index = np.random.choice(self.size, size=batch_size, replace=False)

        # Convert sampled data to Tensors and move directly to the target device (GPU)
        s = torch.tensor(self.s[index], dtype=torch.float).to(self.device)
        priv = torch.tensor(
            self.priv[index], dtype=torch.float).to(self.device)

        # Add channel dimension to depth maps -> (B, 1, H, W) to match CNN input format
        wrist_d = torch.tensor(
            self.wrist_d[index], dtype=torch.float).unsqueeze(1).to(self.device)
        global_d = torch.tensor(
            self.global_d[index], dtype=torch.float).unsqueeze(1).to(self.device)

        a = torch.tensor(self.a[index], dtype=torch.float).to(self.device)
        r = torch.tensor(self.r[index], dtype=torch.float).to(self.device)

        s_ = torch.tensor(self.s_[index], dtype=torch.float).to(self.device)
        priv_ = torch.tensor(
            self.priv_[index], dtype=torch.float).to(self.device)
        wrist_d_ = torch.tensor(
            self.wrist_d_[index], dtype=torch.float).unsqueeze(1).to(self.device)
        global_d_ = torch.tensor(
            self.global_d_[index], dtype=torch.float).unsqueeze(1).to(self.device)

        dw = torch.tensor(self.dw[index], dtype=torch.float).to(self.device)

        # Return all tensors required for calculating Target_Q and Actor_Loss
        return s, priv, wrist_d, global_d, a, r, s_, priv_, wrist_d_, global_d_, dw
