"""T1: synthetic key-value world.

Sequences are X1 Y1 X2 Y2 ... of random tokens. Token ids 0..n_x-1 are keys (X),
n_x..n_x+n_y-1 are values (Y). Each key has one true value. Keys are drawn with Zipfian
frequency, and each shown value is replaced by a uniformly random Y with probability
``label_noise``. Because we control the data, per-key frequency and noise are known
exactly, which is what the heuristic adversary is allowed to see.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class KVWorldConfig:
    n_x: int = 500
    n_y: int = 500
    zipf_alpha: float = 1.0  # 0 = uniform key frequency
    label_noise: float = 0.1  # P(shown Y is random) per occurrence
    pairs_per_seq: int = 32
    seed: int = 0
    eval_frac: float = 0.5  # fraction of keys held out from the game (student AND adversary)


class KVWorld:
    def __init__(self, cfg: KVWorldConfig):
        self.cfg = cfg
        rng = np.random.default_rng(cfg.seed)
        self.rng = rng
        self.true_value = rng.integers(0, cfg.n_y, size=cfg.n_x)  # index into Y
        # Zipf over a random permutation so frequency rank is unrelated to key id.
        ranks = rng.permutation(cfg.n_x) + 1
        w = ranks.astype(float) ** (-cfg.zipf_alpha)
        self.key_prob = w / w.sum()
        # Key-level split. The game (student confidence, proposer, adversary) only ever sees
        # train_keys; all metrics are reported on eval_keys. Otherwise either player can win
        # by memorising which keys the student gets right, which is not introspection.
        perm = rng.permutation(cfg.n_x)
        n_hold = int(round(cfg.eval_frac * cfg.n_x))
        self.eval_keys = np.sort(perm[:n_hold])
        self.train_keys = np.sort(perm[n_hold:])
        self.key_counts = np.zeros(cfg.n_x, dtype=np.int64)  # occurrences seen in training

    def state_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if k != "rng"}

    @classmethod
    def from_state_dict(cls, state: dict) -> "KVWorld":
        world = cls.__new__(cls)
        world.__dict__.update(state)
        world.rng = np.random.default_rng(state["cfg"].seed + 1)
        return world

    @property
    def vocab_size(self) -> int:
        return self.cfg.n_x + self.cfg.n_y

    def y_token(self, y_idx):
        return y_idx + self.cfg.n_x

    def sample_batch(self, batch_size: int, track_counts: bool = True) -> torch.Tensor:
        """Returns LongTensor [batch, 2 * pairs_per_seq] of interleaved X, Y tokens."""
        c = self.cfg
        keys = self.rng.choice(c.n_x, size=(batch_size, c.pairs_per_seq), p=self.key_prob)
        vals = self.true_value[keys]
        noisy = self.rng.random(keys.shape) < c.label_noise
        vals = np.where(noisy, self.rng.integers(0, c.n_y, size=keys.shape), vals)
        if track_counts:
            np.add.at(self.key_counts, keys.ravel(), 1)
        seq = np.empty((batch_size, 2 * c.pairs_per_seq), dtype=np.int64)
        seq[:, 0::2] = keys
        seq[:, 1::2] = self.y_token(vals)
        return torch.from_numpy(seq)

    def heuristic_features(self, keys: np.ndarray) -> np.ndarray:
        """What the T1 adversary may see: log training frequency of the key, and noise rate.

        Noise is global here (one feature value), but kept as a column so per-key noise
        can be added later without changing adversary code.
        """
        keys = np.asarray(keys)
        return np.stack(
            [np.log1p(self.key_counts[keys]), np.full(len(keys), self.cfg.label_noise)], axis=1
        )
