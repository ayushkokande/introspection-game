# Roadmap

Ordered by what has to be true before the next step is worth running.

## First numbers (configs/t1_base.yaml, seed 0, CPU)

Held-out keys, 250 of them, student accuracy 0.576.

| Predictor | AUROC | Brier | Reward gap vs. frequency adversary |
|---|---|---|---|
| Frequency adversary (binned) | 0.819 | 0.167 | 0 |
| Probe on answer logits | 0.839 | 0.159 | +0.008 |
| Probe on answer logits + frequency | 0.873 | 0.143 | +0.024 |
| Probe on raw residual | 0.534 | 0.246 | −0.079 |
| Game-trained confidence head (`logits` input, 2000 steps) | 0.788 | | −0.023 (in-game reward +0.12) |

Readings:
- There is a residual to find: internal logit confidence carries information frequency does not.
- The game head does not yet reach the logit-probe ceiling, and in-game reward badly overstates held-out performance.
- A linear probe on the raw residual is useless at this data size; the per-key embedding dominates it.

## 1. T1, done rigorously

- [ ] Close the gap between the game head and the logit-probe ceiling: learning rate, steps, weight decay, Brier vs. log.
- [ ] Residual check (`scripts/t1.py sanity`): sweep `d_model`, `zipf_alpha`, `label_noise` and record where internal
      probes beat the frequency adversary on held-out keys.
- [ ] `conf_input: resid` and `both`: can the head learn to read stored knowledge instead of memorising key identity?
- [ ] Game vs. ablations: full game; `train_proposer: false`; log vs. Brier rule.
- [ ] Exploitability: after training, fit a best-response adversary on the final proposal distribution and report the gap.
- [ ] Self vs. other control: train a second memoriser (different seed), have this student's confidence head predict *its*
      correctness. If it does as well as on itself, the head is reading question difficulty, not self-knowledge.
- [ ] Mechanism: activation patching from block 1 into block 2; edit one key's stored value and check only that key's
      confidence moves.
- [ ] Adversary ladder: features only, + answer token, + k sampled answers (needs a stochastic answer path), + another
      model's residual.

## 2. T2, small

- [ ] A few hundred MEMIT edits on fictional entities, including edits that fail on some paraphrases.
- [ ] Dose-response: scale the update, confidence should track edit strength; revert, confidence should drop.
- [ ] Edit-detection shortcut check on the student's confidence features.
- [ ] Comparison arm: same facts injected by synthetic-document fine-tuning.
- [ ] Train on organic, test on injected, and the reverse.

## 3. Section 4, pilot

- [ ] Headroom first: pass@1 vs. pass@k and semantic entropy by size and popularity bucket.
- [ ] One model size; RLVR vs. (A) gold-answer probability vs. (B) introspective confidence.
- [ ] Label-free arm for (B). Lead with selective accuracy and abstention-aware scores.
- [ ] Monitor the confidence head's calibration on fresh samples throughout (co-adaptation hacking).

## Open questions

- Which correctness target in T1 under label noise: clean true value (current) or the noisy-label mode?
- Does block 2 need attention at all for length-1 queries, or is an MLP head on the residual equivalent?
- Model family for T2 and Section 4 (Qwen2.5-3B/7B/14B proposed).
