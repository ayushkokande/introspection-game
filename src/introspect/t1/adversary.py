"""External predictors of student correctness, plus the internal-probe ceiling.

All predictors share one interface: ``fit(features, y)`` then ``predict(features) -> p``.
They must always be fit on ``world.train_keys`` and evaluated on ``world.eval_keys``
(a key-level split), so none of them can win by memorising which keys the student knows.
"""

from __future__ import annotations

import numpy as np
import torch


class BinnedFrequencyAdversary:
    """Nonparametric P(correct | log-frequency bin), with Beta smoothing.

    With global noise, frequency is the only external cue in T1, so a well-binned estimate
    of P(correct | frequency) approximates the Bayes-optimal heuristic adversary.
    """

    def __init__(self, n_bins: int = 10, prior_strength: float = 2.0):
        self.n_bins = n_bins
        self.prior_strength = prior_strength

    def fit(self, feats: np.ndarray, y: np.ndarray):
        f = feats[:, 0]
        self.edges = np.unique(np.quantile(f, np.linspace(0, 1, self.n_bins + 1)))
        base = y.mean()
        idx = self._bin(f)
        self.table = np.full(len(self.edges) + 1, base)
        for b in np.unique(idx):
            m = idx == b
            a = self.prior_strength
            self.table[b] = (y[m].sum() + a * base) / (m.sum() + a)
        return self

    def _bin(self, f):
        return np.searchsorted(self.edges[1:-1], f, side="right")

    def predict(self, feats: np.ndarray) -> np.ndarray:
        return self.table[self._bin(feats[:, 0])]


class LogisticPredictor:
    """L2-regularised logistic regression (torch, full-batch LBFGS).

    Used both as a parametric heuristic adversary (on external features) and as the
    supervised internal probe (on the student's residual stream), which gives the ceiling
    of how much correctness information is linearly available inside the student.
    """

    def __init__(self, l2: float = 1e-2, standardize: bool = True):
        self.l2 = l2
        self.standardize = standardize

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=np.float32)
        self.mu = X.mean(0) if self.standardize else 0.0
        self.sd = X.std(0) + 1e-6 if self.standardize else 1.0
        Xt = torch.from_numpy((X - self.mu) / self.sd)
        yt = torch.from_numpy(np.asarray(y, dtype=np.float32))
        self.w = torch.zeros(X.shape[1], requires_grad=True)
        self.b = torch.zeros(1, requires_grad=True)
        opt = torch.optim.LBFGS([self.w, self.b], max_iter=200, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            logits = Xt @ self.w + self.b
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, yt)
            loss = loss + self.l2 * (self.w**2).sum()
            loss.backward()
            return loss

        opt.step(closure)
        return self

    @torch.no_grad()
    def predict(self, X: np.ndarray) -> np.ndarray:
        Xt = torch.from_numpy((np.asarray(X, dtype=np.float32) - self.mu) / self.sd)
        return torch.sigmoid(Xt @ self.w + self.b).numpy()
