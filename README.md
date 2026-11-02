# Knowing What You Know

An adversarial RL game for introspective confidence.
Course project for CSCI-GA 3033-125 (Special Topics: RL with Foundation Models), Ayush Kokande.

A model's stated confidence can be calibrated for the wrong reason, for example because a fact is popular.
The game rewards a student model only for the part of its confidence that an external adversary cannot reproduce:

```
R_student = S(c_s, y) - S(c_a, y)      S = proper scoring rule (negative Brier or log)
```

Calibration an outsider can match earns zero, so positive reward is privileged self-knowledge.

## Research questions

- **RQ1.** Can RL train confidence that predicts the model's own correctness better than the best heuristic predictor?
- **RQ2.** Does it generalise to knowledge injected after pretraining (ROME / MEMIT edits)?
- **RQ3.** Used as a reward, does introspective confidence improve benchmark accuracy, especially for small models?

## Status

| Part | Status |
|---|---|
| Scoring rules, reward gap, metrics (AUROC, ECE, Brier decomposition) | done, tested |
| T1 synthetic key-value world, frozen-answer student, memoriser | done, tested |
| T1 week-1 residual check (probe ceiling vs. heuristic adversary) | done |
| T1 game (confidence head + REINFORCE proposer + refit adversary) | first version, runs; does not yet beat the adversary on held-out keys |
| T2 knowledge editing (MEMIT, dose-response, edit-detection diagnostic) | planned, see `src/introspect/t2` |
| Section 4 benchmarks (headroom check, rewards A/B, baselines) | planned, see `src/introspect/bench` |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"          # T1 only needs torch, numpy, pyyaml
pip install -e ".[llm]"          # later: T2 and Section 4
pytest
```

## Running T1

```bash
# check the pipeline end to end (seconds)
python scripts/t1.py memorize --config configs/t1_smoke.yaml
python scripts/t1.py sanity   --config configs/t1_smoke.yaml
python scripts/t1.py game     --config configs/t1_smoke.yaml

# proposal defaults, 500 X / 500 Y tokens (about a minute on a laptop CPU)
python scripts/t1.py memorize --config configs/t1_base.yaml
python scripts/t1.py sanity   --config configs/t1_base.yaml
python scripts/t1.py game     --config configs/t1_base.yaml
```

Outputs go to `runs/<run_name>/`.

**Run `sanity` before trusting any game result.** It fits a supervised linear probe on the student's residual stream (the ceiling)
and the best frequency-based adversary on train keys, then scores both on held-out keys.
If the probe does not beat the adversary, frequency and noise already determine what is stored and the game has nothing to find.
Turn the capacity (`student.d_mlp`), Zipf exponent and noise knobs until there is a gap.

## Design decisions that guard against known failure modes

- **Frozen answer path.** The student controls `y` through its answer. With a trainable answer, it can win by answering wrong
  exactly where the adversary predicts "correct" (see `test_sandbagging_incentive_exists_when_answer_is_trainable`).
  In T1, block 1 and the unembedding are frozen after memorisation; block 2 and the confidence head read a detached residual
  and never feed the answer logits.
- **Key-level split.** The whole game (student, proposer, adversary) sees only `train_keys`; all metrics are on `eval_keys`.
  Otherwise either player can win by memorising which keys the student gets right.
- **Fresh adversary at evaluation.** In-game reward is measured against an adversary the proposer has been exploiting.
  Evaluation refits an adversary from scratch on train keys and scores on held-out keys.
- **Frozen random embeddings, small width.** Learned per-key embeddings act as a lookup table, and at `d_model: 64` all
  500 keys are memorised perfectly, leaving nothing to predict. The base config (`d_model: 16`) lands at ~57% accuracy.
- **Confidence-head input (`student.conf_input`).** Reading the raw residual (`resid`) lets the head memorise correctness
  per key, because the residual carries each key's unique embedding; it overfits train keys and fails on held-out keys.
  The default `logits` reads the sorted top-k log-probs of the student's own answer distribution, which cannot encode key
  identity. Getting `resid` to generalise is itself an experiment.
- **Proposer exploration floor.** Without `game.uniform_mix`, the proposer collapses onto a few keys within a few hundred
  steps and the head learns nothing general.
- **Bounded log score.** Probabilities are clipped so log-score differences stay finite.
- **What the adversary term does.** `S(c_a, y)` does not depend on `c_s`, so for fixed questions it does not change the
  confidence gradient. It acts through question proposal. Run with `game.train_proposer: false` as the ablation.

## Layout

```
configs/                 YAML configs (t1_base = proposal defaults, t1_smoke = quick check)
scripts/t1.py            T1 entry point: memorize | sanity | game
src/introspect/
  scoring.py             proper scoring rules, reward gap, AUROC / ECE / Brier decomposition
  utils.py               config loading, seeding, JSON output
  t1/data.py             synthetic key-value world, Zipfian frequency, label noise, key split
  t1/model.py            student: frozen answer path (block 1) + introspection path (block 2 + head)
  t1/memorizer.py        stage 1 training, per-key correctness, checkpoints
  t1/adversary.py        binned-frequency and logistic adversaries; logistic probe
  t1/sanity.py           week-1 residual check
  t1/game.py             the introspection game
  t2/                    knowledge-editing testbed (planned)
  bench/                 Section 4 benchmark experiments (planned)
docs/roadmap.md          experiment plan and open questions
tests/
```
