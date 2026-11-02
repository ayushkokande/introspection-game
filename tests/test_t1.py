import numpy as np
import torch

from introspect.t1.data import KVWorld, KVWorldConfig
from introspect.t1.game import GameConfig, IntrospectionGame
from introspect.t1.memorizer import MemorizerConfig, build, train_memorizer
from introspect.t1.model import StudentConfig
from introspect.t1.sanity import residual_check


def tiny():
    torch.manual_seed(0)
    wcfg = KVWorldConfig(n_x=40, n_y=40, pairs_per_seq=8, seed=0)
    world, model = build(wcfg, StudentConfig(d_model=16, n_heads=2, d_mlp=16, conf_hidden=16))
    train_memorizer(world, model, MemorizerConfig(steps=60, batch_size=32, log_every=1000), log=lambda *_: None)
    return world, model


def test_world_layout_and_split():
    w = KVWorld(KVWorldConfig(n_x=50, n_y=30, pairs_per_seq=4))
    seq = w.sample_batch(3)
    assert seq.shape == (3, 8)
    assert (seq[:, 0::2] < 50).all() and (seq[:, 1::2] >= 50).all() and (seq[:, 1::2] < 80).all()
    assert len(np.intersect1d(w.train_keys, w.eval_keys)) == 0
    assert len(w.train_keys) + len(w.eval_keys) == 50
    assert w.key_counts.sum() == 12


def test_answer_path_frozen_during_game():
    world, model = tiny()
    before = {k: v.clone() for k, v in model.state_dict().items() if not k.startswith(("block2", "conf_head"))}
    game = IntrospectionGame(model, world, GameConfig(steps=5, batch_size=16, refit_every=2, eval_every=5))
    game.run(log=lambda *_: None)
    for k, v in model.state_dict().items():
        if k in before:
            assert torch.equal(v, before[k]), k


def test_sanity_report_runs():
    world, model = tiny()
    rep = residual_check(model, world)
    assert {"probe_resid", "probe_logit", "adv_binned_freq"} <= set(rep["metrics"])
    assert all(np.isfinite(g) for g in rep["gaps_vs_adv_binned_freq"].values())
