"""Week-1 check: is there any private residual for the game to find?

Compares, on held-out keys:
  * external predictors fit on heuristic features (frequency, noise), and
  * a supervised linear probe on the student's own residual stream (the ceiling).

If the probe does not beat the best external predictor, frequency and noise already
determine what is stored and the reward gap is zero by design. Tune capacity
(d_mlp), Zipf exponent and noise until it does.
"""

from __future__ import annotations

import numpy as np
import torch

from ..scoring import reward_gap, summarize
from .adversary import BinnedFrequencyAdversary, LogisticPredictor
from .data import KVWorld
from .memorizer import key_correctness
from .model import Student, query_keys


@torch.no_grad()
def internal_features(model: Student, world: KVWorld) -> dict[str, np.ndarray]:
    """Student-internal views of each key: the raw residual, and the answer's logit confidence
    (max softmax probability over Y and the top-1 minus top-2 logit margin)."""
    _, resid, _ = query_keys(model, torch.arange(world.cfg.n_x), world.cfg.n_x)
    logits = model.unembed(model.ln_f(resid))[:, 0, world.cfg.n_x :]
    top2 = logits.topk(2, dim=-1).values
    logit_feats = torch.stack(
        [logits.softmax(-1).max(-1).values, top2[:, 0] - top2[:, 1]], dim=1
    )
    return {"resid": resid[:, 0].numpy(), "logit": logit_feats.numpy()}


def residual_check(model: Student, world: KVWorld, rule: str = "brier") -> dict:
    """Fit every predictor on train keys, score on eval keys.

    External predictors see only frequency (and noise); internal ones see the student's
    activations. ``gaps`` is the mean reward gap of each internal predictor against the
    binned-frequency adversary, i.e. how much the game could reward in principle.
    """
    y = key_correctness(model, world)
    tr, ev = world.train_keys, world.eval_keys
    freq = world.heuristic_features(np.arange(world.cfg.n_x))
    internal = internal_features(model, world)

    # name -> (predictor, feature matrix over all keys)
    spec = {
        "adv_binned_freq": (BinnedFrequencyAdversary(), freq),
        "adv_logistic_freq": (LogisticPredictor(), freq[:, :1]),
        "probe_logit": (LogisticPredictor(), internal["logit"]),
        "probe_logit+freq": (LogisticPredictor(), np.concatenate([internal["logit"], freq[:, :1]], 1)),
        "probe_resid": (LogisticPredictor(l2=1e-1), internal["resid"]),
        "probe_resid+freq": (
            LogisticPredictor(l2=1e-1),
            np.concatenate([internal["resid"], freq[:, :1]], 1),
        ),
    }
    preds = {"base_rate": np.full(len(ev), y[tr].mean())}
    degenerate = len(np.unique(y[tr])) < 2
    for name, (m, X) in spec.items():
        preds[name] = preds["base_rate"] if degenerate else m.fit(X[tr], y[tr]).predict(X[ev])

    report = {
        "accuracy_all_keys": float(y.mean()),
        "accuracy_eval_keys": float(y[ev].mean()),
        "n_train_keys": int(len(tr)),
        "n_eval_keys": int(len(ev)),
        "metrics": {k: summarize(p, y[ev]) for k, p in preds.items()},
        "gaps_vs_adv_binned_freq": {
            k: float(np.mean(reward_gap(preds[k], preds["adv_binned_freq"], y[ev], rule)))
            for k in preds
            if k.startswith("probe")
        },
    }
    if degenerate:
        report["warning"] = "student is always right (or always wrong) on train keys; lower capacity or raise zipf_alpha"
    return report
