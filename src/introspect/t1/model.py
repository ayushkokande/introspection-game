"""T1 student: a two-block transformer with a frozen answer path.

Block 1 (+ embedding and unembedding) is trained on memorisation, then frozen. All stored
knowledge lives there by construction. Block 2 reads block 1's residual stream and feeds
only the confidence head; it never touches the answer logits. That keeps the answer fixed
during the game, which closes the sandbagging exploit (answering wrong where the adversary
predicts "correct").
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class StudentConfig:
    vocab_size: int = 1000
    n_x: int = 500  # Y tokens start here; set by build()
    d_model: int = 16  # main capacity knob
    n_heads: int = 2
    d_mlp: int = 32
    max_len: int = 64
    conf_hidden: int = 64
    # What the confidence head reads:
    #   "logits": sorted top-k log-probs of the student's own answer distribution
    #             (permutation-invariant, so it cannot encode key identity)
    #   "resid":  block 2 over the raw residual stream. Harder: the residual carries a unique
    #             embedding per key, so the head can memorise correctness per key instead of
    #             reading what is stored. Whether it learns not to is a research question.
    #   "both":   concatenation of the two
    conf_input: str = "logits"
    conf_topk: int = 8
    # Fix token/position embeddings at random init. Otherwise a learned embedding per key is a
    # lookup table that bypasses the MLP capacity limit and memorisation is trivially perfect.
    freeze_embeddings: bool = True


class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_mlp: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(nn.Linear(d_model, d_mlp), nn.GELU(), nn.Linear(d_mlp, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.shape[1]
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), 1)
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, attn_mask=mask, need_weights=False)
        x = x + a
        return x + self.mlp(self.ln2(x))


class Student(nn.Module):
    def __init__(self, cfg: StudentConfig):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos = nn.Embedding(cfg.max_len, cfg.d_model)
        self.block1 = Block(cfg.d_model, cfg.n_heads, cfg.d_mlp)
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.unembed = nn.Linear(cfg.d_model, cfg.vocab_size)
        # Introspection side: only trained by the game.
        assert cfg.conf_input in ("logits", "resid", "both"), cfg.conf_input
        self.block2 = Block(cfg.d_model, cfg.n_heads, cfg.d_mlp)
        d_in = {"logits": cfg.conf_topk, "resid": cfg.d_model, "both": cfg.conf_topk + cfg.d_model}
        self.conf_head = nn.Sequential(
            nn.LayerNorm(d_in[cfg.conf_input]),
            nn.Linear(d_in[cfg.conf_input], cfg.conf_hidden),
            nn.GELU(),
            nn.Linear(cfg.conf_hidden, 1),
        )
        if cfg.freeze_embeddings:
            self.tok.weight.requires_grad_(False)
            self.pos.weight.requires_grad_(False)

    # -- answer path -------------------------------------------------------------
    def residual1(self, seq: torch.Tensor) -> torch.Tensor:
        pos = torch.arange(seq.shape[1], device=seq.device)
        return self.block1(self.tok(seq) + self.pos(pos))

    def answer_logits(self, seq: torch.Tensor) -> torch.Tensor:
        return self.unembed(self.ln_f(self.residual1(seq)))

    # -- confidence path ---------------------------------------------------------
    def confidence(self, seq: torch.Tensor, resid1: torch.Tensor | None = None) -> torch.Tensor:
        """P(answer at each position is correct), shape [B, T]. Answer path is detached."""
        if resid1 is None:
            resid1 = self.residual1(seq)
        resid1 = resid1.detach()
        feats = []
        if self.cfg.conf_input in ("logits", "both"):
            with torch.no_grad():
                logp = self.unembed(self.ln_f(resid1))[..., self.cfg.n_x :].log_softmax(-1)
                feats.append(logp.topk(self.cfg.conf_topk, dim=-1).values)
        if self.cfg.conf_input in ("resid", "both"):
            feats.append(self.block2(resid1))
        return torch.sigmoid(self.conf_head(torch.cat(feats, -1)).squeeze(-1))

    def freeze_answer_path(self) -> None:
        for m in (self.tok, self.pos, self.block1, self.ln_f, self.unembed):
            for p in m.parameters():
                p.requires_grad_(False)

    def introspection_parameters(self):
        params = list(self.conf_head.parameters())
        if self.cfg.conf_input != "logits":
            params += list(self.block2.parameters())
        return params


def memorisation_loss(model: Student, seq: torch.Tensor) -> torch.Tensor:
    """Cross-entropy on predicting each Y from the sequence up to its X (Y positions only)."""
    logits = model.answer_logits(seq[:, :-1])
    targets = seq[:, 1:]
    # positions 0, 2, 4, ... of the shifted input are X tokens whose next token is a Y
    logits, targets = logits[:, 0::2], targets[:, 0::2]
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))


@torch.no_grad()
def query_keys(model: Student, keys: torch.Tensor, n_x: int):
    """Ask the student about single keys. Returns (answer y-index, resid1 at the key, seq).

    The answer is restricted to Y tokens. Queries are length-1 sequences [X], so the
    student must answer from stored knowledge, not from in-context copies.
    """
    seq = keys.view(-1, 1)
    resid = model.residual1(seq)
    logits = model.unembed(model.ln_f(resid))[:, 0, n_x:]
    return logits.argmax(-1), resid, seq
