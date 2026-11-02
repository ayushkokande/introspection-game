"""Stage 2 of T1: the introspection game.

Players
  * student confidence head (block 2 + head), reads the frozen answer path's residual
  * student proposer, a distribution over train keys (which questions to ask itself)
  * adversary, sees only external features, refit periodically on recent proposals

In T1 the confidence is a scalar, so it is trained by direct gradient on the reward gap.
GRPO is only needed once proposals and confidences are sampled text (T2 / Section 4).
The proposer is discrete and is trained with REINFORCE, a group-mean baseline (as in GRPO)
and an entropy bonus.

The adversary term in the reward does not depend on c_s, so it does not change the
confidence gradient; it acts only through the proposer. Report the in-game reward, but
treat the held-out evaluation against a freshly fit adversary as the real result.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
import torch

from ..scoring import reward_gap, summarize
from .adversary import BinnedFrequencyAdversary, LogisticPredictor
from .data import KVWorld
from .memorizer import key_correctness
from .model import Student


@dataclass
class GameConfig:
    steps: int = 2000
    batch_size: int = 64  # questions per step (one group for the proposer baseline)
    lr_conf: float = 1e-3
    weight_decay_conf: float = 0.0
    lr_proposer: float = 5e-3
    entropy_coef: float = 0.05
    uniform_mix: float = 0.5  # fraction of each batch drawn uniformly (exploration floor)
    scoring_rule: str = "brier"  # "brier" or "log"
    adversary: str = "binned"  # "binned" or "logistic"
    refit_every: int = 100
    refit_window: int = 4096  # number of recent proposals the adversary is refit on
    train_proposer: bool = True  # False = uniform questions (ablation)
    eval_every: int = 250


def make_adversary(kind: str):
    return BinnedFrequencyAdversary() if kind == "binned" else LogisticPredictor()


class IntrospectionGame:
    def __init__(self, model: Student, world: KVWorld, cfg: GameConfig):
        self.model, self.world, self.cfg = model, world, cfg
        self.y = key_correctness(model, world)  # fixed: the answer path is frozen
        self.feats = world.heuristic_features(np.arange(world.cfg.n_x))
        self.train_keys = torch.as_tensor(world.train_keys)
        self.proposer_logits = torch.zeros(len(world.train_keys), requires_grad=True)
        self.opt_conf = torch.optim.AdamW(
            model.introspection_parameters(), lr=cfg.lr_conf, weight_decay=cfg.weight_decay_conf
        )
        self.opt_prop = torch.optim.Adam([self.proposer_logits], lr=cfg.lr_proposer)
        self.history: deque[int] = deque(maxlen=cfg.refit_window)
        # Initial adversary: fit on uniformly sampled train keys.
        self.adversary = make_adversary(cfg.adversary).fit(
            self.feats[world.train_keys], self.y[world.train_keys]
        )

    # ------------------------------------------------------------------ game step
    def step(self) -> dict:
        cfg = self.cfg
        dist = torch.distributions.Categorical(logits=self.proposer_logits)
        n_prop = int(round((1 - cfg.uniform_mix) * cfg.batch_size)) if cfg.train_proposer else 0
        idx = torch.cat(
            [
                dist.sample((n_prop,)),
                torch.randint(len(self.train_keys), (cfg.batch_size - n_prop,)),
            ]
        )
        keys = self.train_keys[idx]
        self.history.extend(keys.tolist())

        k_np = keys.numpy()
        y = torch.from_numpy(self.y[k_np])
        c_a = torch.from_numpy(self.adversary.predict(self.feats[k_np])).float()
        c_s = self.model.confidence(keys.view(-1, 1))[:, 0]

        r = reward_gap(c_s, c_a, y, cfg.scoring_rule)
        self.opt_conf.zero_grad()
        (-r.mean()).backward()
        self.opt_conf.step()

        out = {"reward": r.mean().item(), "proposer_entropy": dist.entropy().item()}
        if n_prop > 0:
            r_d = r.detach()[:n_prop]  # proposer is credited only for its own samples
            adv = r_d - r_d.mean()  # group baseline
            loss_p = -(adv * dist.log_prob(idx[:n_prop])).mean() - cfg.entropy_coef * dist.entropy()
            self.opt_prop.zero_grad()
            loss_p.backward()
            self.opt_prop.step()
        return out

    def refit_adversary(self) -> None:
        keys = np.fromiter(self.history, dtype=np.int64)
        if len(np.unique(self.y[keys])) < 2:
            return
        self.adversary = make_adversary(self.cfg.adversary).fit(self.feats[keys], self.y[keys])

    # ------------------------------------------------------------------ evaluation
    @torch.no_grad()
    def evaluate(self) -> dict:
        """Held-out keys, uniform distribution, adversary refit from scratch on train keys."""
        w = self.world
        ev = w.eval_keys
        y_ev = self.y[ev]
        c_s = self.model.confidence(torch.as_tensor(ev).view(-1, 1))[:, 0].numpy()
        fresh = make_adversary(self.cfg.adversary).fit(self.feats[w.train_keys], self.y[w.train_keys])
        c_a = fresh.predict(self.feats[ev])
        return {
            "eval_reward_gap": float(np.mean(reward_gap(c_s, c_a, y_ev, self.cfg.scoring_rule))),
            "student": summarize(c_s, y_ev),
            "fresh_adversary": summarize(c_a, y_ev),
        }

    def run(self, log=print) -> list[dict]:
        logs = []
        for t in range(1, self.cfg.steps + 1):
            out = self.step()
            if t % self.cfg.refit_every == 0:
                self.refit_adversary()
            if t % self.cfg.eval_every == 0 or t == self.cfg.steps:
                ev = self.evaluate()
                out.update(step=t, **ev)
                logs.append(out)
                log(
                    f"[game] step {t:5d}  in-game R {out['reward']:+.4f}  "
                    f"eval gap {ev['eval_reward_gap']:+.4f}  "
                    f"AUROC student {ev['student']['auroc']:.3f} vs adv {ev['fresh_adversary']['auroc']:.3f}  "
                    f"H(prop) {out['proposer_entropy']:.2f}"
                )
        return logs
