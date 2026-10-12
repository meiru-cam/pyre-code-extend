"""Pause-token training loss terms, transcribed from the paper's definitions.

Upstream:   paper only (no official code). Goyal, Ji, Rawat, Menon, Kumar,
            Nagarajan, "Think before you speak: Training Language Models With
            Pause Tokens", ICLR 2024. https://arxiv.org/abs/2310.02226
Commit:     arXiv 2310.02226v3 (revised 21 Apr 2024)
Files:      Section 3.1 "Learning and inference with the <pause> token":
            "Pretraining with the <pause> token" (the S_ignore set and Equation 3)
            and "Finetuning with the <pause> token"; Appendix, Algorithm 1
            (Pause-pretraining) and Algorithm 2 (Pause-finetuning).
Symbols:    S_ignore, L_PausePT (Equation 3), L_PauseFT (Algorithm 2)
Retrieved:  2026-10-12
License:    n/a: transcribed from a formula. No text or code is copied; the
            paper itself is distributed under the arXiv non-exclusive
            distribution license 1.0.

WHY THIS FILE EXISTS
--------------------
The paper released no code, so pause_token_training_mask has no upstream
implementation to vendor. The user decided this exercise may use a paper-only
oracle. This module writes the paper's definitions down in the paper's own
indexing (1-based tokens; position k is the prefix f(p_1:k) that predicts token
k + 1), with no torch, no -100 sentinel and no HF label shift. The oracle test
converts the reference solution's HF-style labels into the same (k, token)
loss terms and compares, so a label-convention or off-by-one error in the
reference shows up as a mismatch instead of being hidden by a shared mistake.

The transcribed definitions, verbatim in plain text:

  S_ignore = { k : p~_{k+1} = <pause> }                          (Section 3.1)
  L_PausePT(f, p~_{1:N+M}) = sum over k = 1 .. N+M-1, k not in S_ignore,
                             of CE(p~_{k+1}, f(p~_{1:k}))         (Equation 3)
  p~_{1:N+M_ft} = Concat[p_{1:N}, [<pause>] x M_ft]              (Algorithm 2)
  L_PauseFT = sum over k = 0 .. T-1 of
              CE(t_{k+1}, f(Concat[p~_{1:N+M_ft}, t_{1:k}]))      (Algorithm 2)

WHAT WAS CHANGED, AND WHY
-------------------------
  * The paper gives math, not code. Each function below is one definition, in
    plain Python lists of ints, so it cannot share a bug with a torch solution.
  * ``pause_finetune_loss_terms`` combines the two losses the way the exercise
    does: the finetuning sum of Algorithm 2 restricted, as Equation 3 does, to
    positions outside S_ignore. The paper only states S_ignore for pretraining;
    applying it during finetuning changes nothing unless a target token is
    itself the pause id, and the exercise pins that case.
  * Algorithm 1 writes k in [0, N+M-1] while Equation 3 sums from k = 1; the
    two agree because position 0 has no prefix to predict from. Positions here
    run from 1, as in Equation 3.

NOTHING HERE IMPORTS A TASK SOLUTION. This module must stay independent of the
answers it is used to check.
"""

from __future__ import annotations

__all__ = [
    "pause_finetune_loss_terms",
    "pause_finetune_sequence",
    "pause_pretrain_loss_terms",
    "s_ignore",
]


def pause_finetune_sequence(prefix: list[int], target: list[int], pause: int, m_ft: int) -> list[int]:
    """Algorithm 2: Concat[p_{1:N}, [<pause>] x M_ft], followed by the target t_{1:T}."""
    return list(prefix) + [pause] * m_ft + list(target)


def s_ignore(seq: list[int], pause: int) -> set[int]:
    """Section 3.1: the 1-based positions k whose next token p~_{k+1} is <pause>."""
    return {k for k in range(1, len(seq)) if seq[k + 1 - 1] == pause}


def pause_pretrain_loss_terms(seq: list[int], pause: int) -> list[tuple[int, int]]:
    """Equation 3: the (k, p~_{k+1}) pairs that enter L_PausePT, for k = 1 .. len - 1."""
    ignore = s_ignore(seq, pause)
    return [(k, seq[k + 1 - 1]) for k in range(1, len(seq)) if k not in ignore]


def pause_finetune_loss_terms(
    prefix: list[int], target: list[int], pause: int, m_ft: int
) -> list[tuple[int, int]]:
    """Algorithm 2's loss, minus the positions in S_ignore.

    The term for t_{j+1} (j = 0 .. T-1) is predicted from the first N + M_ft + j
    tokens, so its position is k = N + M_ft + j.
    """
    seq = pause_finetune_sequence(prefix, target, pause, m_ft)
    ignore = s_ignore(seq, pause)
    start = len(prefix) + m_ft
    terms = [(start + j, target[j]) for j in range(len(target))]
    return [(k, token) for k, token in terms if k not in ignore]
