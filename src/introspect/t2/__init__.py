"""T2: knowledge-editing testbed (ROME / MEMIT facts about fictional entities).

Not implemented yet. Planned modules:

  facts.py     generate fictional-entity facts with paraphrase sets (train / test paraphrases)
  edit.py      apply MEMIT batches; also a synthetic-document fine-tuning arm for comparison
  dose.py      scale the rank-one update to get a dose-response curve; revert edits
  diagnose.py  linear "edited vs organic" classifier on hidden states AND on the student's
               confidence features (shortcut check)
  game.py      text version of the game: GRPO on verbalised confidence, answer from a frozen
               copy of the model so the student cannot sandbag

Install the LLM extras first:  pip install -e ".[llm]"
"""
