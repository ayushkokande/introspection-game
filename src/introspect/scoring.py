"""Proper scoring rules, the game reward, and evaluation metrics.

Conventions: every score is "higher is better", so the negative Brier score is used.
Probabilities are confidences that the student's answer is correct; ``y`` is 0/1 correctness
(or a fractional accuracy p_hat over k samples, which both rules accept).
"""

from __future__ import annotations

import numpy as np
import torch

EPS = 1e-4  # clip probabilities so log scores (and their differences) stay bounded


def _clip(p):
    if isinstance(p, torch.Tensor):
        return p.clamp(EPS, 1 - EPS)
    return np.clip(p, EPS, 1 - EPS)


def log_score(p, y):
    """Log score: y*log p + (1-y)*log(1-p). Works on numpy arrays or torch tensors."""
    p = _clip(p)
    lib = torch if isinstance(p, torch.Tensor) else np
    return y * lib.log(p) + (1 - y) * lib.log(1 - p)


def neg_brier(p, y):
    """Negative Brier score: -(p - y)^2."""
    return -((p - y) ** 2)


SCORING_RULES = {"log": log_score, "brier": neg_brier}


def reward_gap(c_student, c_adversary, y, rule: str = "brier"):
    """R_student = S(c_s, y) - S(c_a, y).

    Calibration an outsider can match earns ~0; positive reward means the student's
    confidence carries information the adversary does not have.

    Note: S(c_a, y) does not depend on c_s, so for a *fixed* question and a *fixed* answer
    this term is only a baseline for the confidence update. It shapes behaviour through
    (i) question proposal and (ii) the answer, if the answer is trainable. Keep the answer
    path frozen during the game, or the student can win by answering wrong exactly where
    the adversary expects it to be right (sandbagging).
    """
    S = SCORING_RULES[rule]
    return S(c_student, y) - S(c_adversary, y)


# ---------------------------------------------------------------------------- metrics


def auroc(scores: np.ndarray, y: np.ndarray) -> float:
    """Rank-based AUROC (Mann-Whitney U), ties get average rank. NaN if y is constant."""
    scores = np.asarray(scores, dtype=float)
    y = np.asarray(y).astype(bool)
    n_pos, n_neg = y.sum(), (~y).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    sorted_scores = scores[order]
    i = 0
    while i < len(scores):
        j = i
        while j + 1 < len(scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def ece(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    """Expected calibration error with equal-width bins."""
    p, y = np.asarray(p, float), np.asarray(y, float)
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            total += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(total)


def brier_decomposition(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> dict:
    """Murphy decomposition: Brier ~= reliability - resolution + uncertainty.

    Resolution is the quantity introspection should raise; reliability is calibration error.
    """
    p, y = np.asarray(p, float), np.asarray(y, float)
    base = y.mean()
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    rel = res = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            w = m.mean()
            rel += w * (p[m].mean() - y[m].mean()) ** 2
            res += w * (y[m].mean() - base) ** 2
    return {
        "brier": float(np.mean((p - y) ** 2)),
        "reliability": float(rel),
        "resolution": float(res),
        "uncertainty": float(base * (1 - base)),
    }


def summarize(p: np.ndarray, y: np.ndarray) -> dict:
    """All headline confidence metrics for one predictor."""
    p, y = np.asarray(p, float), np.asarray(y, float)
    out = {
        "auroc": auroc(p, y),
        "ece": ece(p, y),
        "log_score": float(np.mean(log_score(p, y))),
    }
    out.update(brier_decomposition(p, y))
    return out
