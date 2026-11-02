import numpy as np
import torch

from introspect.scoring import auroc, brier_decomposition, ece, log_score, neg_brier, reward_gap


def test_proper_scores_prefer_truth():
    y = np.array([1.0, 0.0])
    for S in (log_score, neg_brier):
        assert S(np.array([0.9, 0.1]), y).sum() > S(np.array([0.6, 0.4]), y).sum()


def test_reward_gap_zero_when_matching_adversary():
    p = np.array([0.2, 0.7, 0.9])
    y = np.array([0.0, 1.0, 1.0])
    for rule in ("brier", "log"):
        assert np.allclose(reward_gap(p, p, y, rule), 0.0)


def test_reward_gap_torch_and_bounded_log():
    c_s = torch.tensor([0.0, 1.0])
    c_a = torch.tensor([1.0, 0.0])
    y = torch.tensor([0.0, 1.0])
    r = reward_gap(c_s, c_a, y, "log")
    assert torch.isfinite(r).all() and (r > 0).all()


def test_sandbagging_incentive_exists_when_answer_is_trainable():
    # Documents why the answer path must be frozen: with a confident adversary,
    # a wrong answer reported with c_s = 0 out-earns a right answer reported with c_s = 1.
    c_a = 0.9
    wrong = reward_gap(np.array([0.0]), np.array([c_a]), np.array([0.0]), "log")
    right = reward_gap(np.array([1.0]), np.array([c_a]), np.array([1.0]), "log")
    assert wrong[0] > right[0]


def test_auroc_basic():
    assert auroc(np.array([0.1, 0.2, 0.8, 0.9]), np.array([0, 0, 1, 1])) == 1.0
    assert auroc(np.array([0.5, 0.5, 0.5, 0.5]), np.array([0, 1, 0, 1])) == 0.5
    assert np.isnan(auroc(np.array([0.1, 0.2]), np.array([1, 1])))


def test_calibration_metrics():
    rng = np.random.default_rng(0)
    p = rng.random(20000)
    y = (rng.random(20000) < p).astype(float)
    assert ece(p, y) < 0.02
    d = brier_decomposition(p, y)
    assert abs(d["brier"] - (d["reliability"] - d["resolution"] + d["uncertainty"])) < 0.01
