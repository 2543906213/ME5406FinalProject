import numpy as np
import torch


class ReplayBuffer:
    """SAC uses circular playback buffer (off-policy)。"""

    def __init__(self, args):
        max_size = int(args.replay_size)
        h = int(args.img_h)
        w = int(args.img_w)

        self.s = np.zeros((max_size, args.state_dim), dtype=np.float32)
        self.s_next = np.zeros((max_size, args.state_dim), dtype=np.float32)

        self.priv = np.zeros((max_size, args.priv_dim), dtype=np.float32)
        self.priv_next = np.zeros((max_size, args.priv_dim), dtype=np.float32)

        self.wrist = np.zeros((max_size, h, w), dtype=np.float32)
        self.wrist_next = np.zeros((max_size, h, w), dtype=np.float32)

        self.global_d = np.zeros((max_size, h, w), dtype=np.float32)
        self.global_d_next = np.zeros((max_size, h, w), dtype=np.float32)

        self.a = np.zeros((max_size, args.action_dim), dtype=np.float32)
        self.r = np.zeros((max_size, 1), dtype=np.float32)
        self.dw = np.zeros((max_size, 1), dtype=np.float32)

        self.max_size = max_size
        self.ptr = 0
        self.size = 0

    def store(
        self,
        s,
        priv,
        wrist,
        global_d,
        a,
        r,
        s_next,
        priv_next,
        wrist_next,
        global_d_next,
        dw,
    ):
        idx = self.ptr

        self.s[idx] = s
        self.priv[idx] = priv
        self.wrist[idx] = wrist
        self.global_d[idx] = global_d

        self.a[idx] = a
        self.r[idx] = r

        self.s_next[idx] = s_next
        self.priv_next[idx] = priv_next
        self.wrist_next[idx] = wrist_next
        self.global_d_next[idx] = global_d_next
        self.dw[idx] = dw

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size: int, device: torch.device):
        idx = np.random.randint(0, self.size, size=batch_size)

        batch = {
            "s": torch.tensor(self.s[idx], dtype=torch.float32, device=device),
            "priv": torch.tensor(self.priv[idx], dtype=torch.float32, device=device),
            "wrist": torch.tensor(self.wrist[idx], dtype=torch.float32, device=device).unsqueeze(1),
            "global_d": torch.tensor(self.global_d[idx], dtype=torch.float32, device=device).unsqueeze(1),
            "a": torch.tensor(self.a[idx], dtype=torch.float32, device=device),
            "r": torch.tensor(self.r[idx], dtype=torch.float32, device=device),
            "s_next": torch.tensor(self.s_next[idx], dtype=torch.float32, device=device),
            "priv_next": torch.tensor(self.priv_next[idx], dtype=torch.float32, device=device),
            "wrist_next": torch.tensor(self.wrist_next[idx], dtype=torch.float32, device=device).unsqueeze(1),
            "global_d_next": torch.tensor(self.global_d_next[idx], dtype=torch.float32, device=device).unsqueeze(1),
            "dw": torch.tensor(self.dw[idx], dtype=torch.float32, device=device),
        }
        return batch
