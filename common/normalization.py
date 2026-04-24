import numpy as np


class RunningMeanStd:
    """Online calculation of mean and standard deviation (Welford's Algorithm)."""

    def __init__(self, shape):
        self.n = 0
        self.mean = np.zeros(shape)
        self.S = np.zeros(shape)
        self.std = np.sqrt(self.S)

    def update(self, x):
        x = np.array(x)
        self.n += 1
        if self.n == 1:
            self.mean = x
            self.std = x
        else:
            old_mean = self.mean.copy()
            self.mean = old_mean + (x - old_mean) / self.n
            self.S = self.S + (x - old_mean) * (x - self.mean)
            self.std = np.sqrt(self.S / self.n)


class Normalization:
    """
    Performs zero-mean unit-variance normalization on input vectors.

    Usage:
        norm = Normalization(shape=30)
        x_normed = norm(x)               # Training: update statistics and normalize
        x_normed = norm(x, update=False) # Evaluation: normalize only, no statistics update
    """

    def __init__(self, shape):
        self.running_ms = RunningMeanStd(shape=shape)

    def __call__(self, x, update=True):
        if update:
            self.running_ms.update(x)
        x = (x - self.running_ms.mean) / (self.running_ms.std + 1e-8)
        return x


class RewardScaling:
    """
    Reward scaling based on discounted returns (divides by standard deviation only, no mean subtraction).

    Step-wise call: r_scaled = r / std(R), where R = γ·R + r is the running discounted return.
    Call reset() at the end of an episode to clear R.
    """

    def __init__(self, gamma: float):
        self.gamma = gamma
        self.running_ms = RunningMeanStd(shape=1)
        self.R = np.zeros(1)

    def __call__(self, r: float) -> float:
        self.R = self.gamma * self.R + r
        self.running_ms.update(self.R)
        return float(r / (self.running_ms.std + 1e-8))

    def reset(self):
        self.R = np.zeros(1)
