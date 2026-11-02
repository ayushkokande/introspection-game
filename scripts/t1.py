"""T1 entry point.

  python scripts/t1.py memorize --config configs/t1_base.yaml   # stage 1: train + freeze answer path
  python scripts/t1.py sanity   --config configs/t1_base.yaml   # week-1 residual check
  python scripts/t1.py game     --config configs/t1_base.yaml   # stage 2: the introspection game

Outputs go to <out_dir>/<run_name>/ (checkpoint, JSON reports).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from introspect.t1.data import KVWorldConfig
from introspect.t1.game import GameConfig, IntrospectionGame
from introspect.t1.memorizer import (
    MemorizerConfig,
    build,
    load_checkpoint,
    save_checkpoint,
    train_memorizer,
)
from introspect.t1.model import StudentConfig
from introspect.t1.sanity import residual_check
from introspect.utils import from_dict, load_yaml, save_json, set_seed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["memorize", "sanity", "game"])
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    run_dir = Path(cfg.get("out_dir", "runs")) / cfg.get("run_name", Path(args.config).stem)
    ckpt = run_dir / "memorizer.pt"
    set_seed(cfg.get("seed", 0))

    if args.stage == "memorize":
        world_cfg = from_dict(KVWorldConfig, cfg.get("world"))
        world, model = build(world_cfg, from_dict(StudentConfig, cfg.get("student")))
        train_memorizer(world, model, from_dict(MemorizerConfig, cfg.get("memorizer")))
        save_checkpoint(ckpt, model, world)
        print(f"saved {ckpt}")
        return

    if not ckpt.exists():
        raise SystemExit(f"{ckpt} not found; run the 'memorize' stage first")
    model, world = load_checkpoint(ckpt)

    if args.stage == "sanity":
        report = residual_check(model, world)
        save_json(report, run_dir / "sanity.json")
        print(f"accuracy (eval keys): {report['accuracy_eval_keys']:.3f}")
        for name, m in report["metrics"].items():
            print(f"  {name:20s} AUROC {m['auroc']:.3f}  Brier {m['brier']:.4f}  "
                  f"resolution {m['resolution']:.4f}")
        for name, g in report["gaps_vs_adv_binned_freq"].items():
            print(f"  reward gap {name:18s} vs frequency adversary: {g:+.4f}")
        if "warning" in report:
            print("WARNING:", report["warning"])
        return

    game = IntrospectionGame(model, world, from_dict(GameConfig, cfg.get("game")))
    logs = game.run()
    save_json(logs, run_dir / "game_log.json")


if __name__ == "__main__":
    main()
