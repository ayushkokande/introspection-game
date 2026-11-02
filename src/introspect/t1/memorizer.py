"""Stage 1 of T1: train the answer path on memorisation, then freeze it."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .data import KVWorld
from .model import Student, StudentConfig, memorisation_loss, query_keys


@dataclass
class MemorizerConfig:
    steps: int = 3000
    batch_size: int = 128
    lr: float = 3e-3
    weight_decay: float = 0.0
    log_every: int = 250


def train_memorizer(world: KVWorld, model: Student, cfg: MemorizerConfig, log=print) -> Student:
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg.steps)
    model.train()
    for step in range(1, cfg.steps + 1):
        seq = world.sample_batch(cfg.batch_size)
        loss = memorisation_loss(model, seq)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step % cfg.log_every == 0 or step == cfg.steps:
            acc = key_correctness(model, world).mean()
            log(f"[memorizer] step {step:5d}  loss {loss.item():.3f}  key-acc {acc:.3f}")
    model.eval()
    model.freeze_answer_path()
    return model


@torch.no_grad()
def key_correctness(model: Student, world: KVWorld) -> np.ndarray:
    """0/1 per key: does the frozen answer path return the key's true value?

    Measured against the clean true value, not the noisy training labels.
    """
    keys = torch.arange(world.cfg.n_x)
    ans, _, _ = query_keys(model, keys, world.cfg.n_x)
    return (ans.numpy() == world.true_value).astype(np.float32)


def save_checkpoint(path: str | Path, model: Student, world: KVWorld) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "model_cfg": model.cfg, "world": world.state_dict()}, path)


def load_checkpoint(path: str | Path) -> tuple[Student, KVWorld]:
    ckpt = torch.load(path, weights_only=False)
    model = Student(ckpt["model_cfg"])
    model.load_state_dict(ckpt["model"])
    model.eval()
    model.freeze_answer_path()
    return model, KVWorld.from_state_dict(ckpt["world"])


def build(world_cfg, student_cfg: StudentConfig) -> tuple[KVWorld, Student]:
    world = KVWorld(world_cfg)
    student_cfg.vocab_size = world.vocab_size
    student_cfg.n_x = world_cfg.n_x
    student_cfg.max_len = max(student_cfg.max_len, 2 * world_cfg.pairs_per_seq)
    return world, Student(student_cfg)
