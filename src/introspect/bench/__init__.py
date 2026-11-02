"""Section 4: from introspective confidence to benchmark gains.

Not implemented yet. Planned modules, in the order they should be built:

  headroom.py  week-1 check: pass@1 vs pass@k and semantic-cluster entropy over 20 samples,
               by model size, correctness and popularity bucket. pass@k - pass@1 bounds the
               accuracy that sharpening can recover.
  datasets.py  PopQA (popularity buckets), TriviaQA, SimpleQA, GSM8K loaders
  rewards.py   RLVR correctness; (A) probability of the gold answer; (B) learned
               introspective confidence; self-certainty and RLCR baselines
  evaluate.py  accuracy, selective accuracy at fixed coverage, abstention-aware score,
               AUROC, ECE, by popularity bucket; matched compute, several seeds

Compare (B) against RLVR at matched labels *and* in the label-free setting
(N labelled for the game, M unlabelled for RL), where (B) has a reason to win.
"""
