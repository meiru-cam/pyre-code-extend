"""RECAP advantage conditioning, transcribed from the pi*0.6 paper's definitions.

Upstream:   paper only (no official code). Physical Intelligence (Amin, Aniceto,
            Balakrishna, Black, ... Zhou), "pi*0.6: a VLA That Learns From
            Experience". https://arxiv.org/abs/2511.14759
Commit:     arXiv 2511.14759v2 (19 Nov 2025)
Files:      Section III (Preliminaries: n-step advantage, improvement
            probability); Section IV-A (distributional value function, Eq. 1 and
            the expected value over bins); Section IV-B (Eq. 2, the improvement
            indicator I_t = 1[A > eps_l], Eq. 3, corrections forced to
            I_t = True); Section V-B ("Advantage: positive" / "Advantage:
            negative" text input); Section V-C (Eq. 5 reward, per-task
            normalization by maximum episode length to (-1, 0)); Section V-D
            (threshold from value-function predictions); Appendix E (Eq. 13,
            CFG with beta > 1); Appendix F (advantage estimation with N = 50 or
            N = T, 30% conditioning dropout, advantage threshold).
Symbols:    r_t (Eq. 5), R_t, V^pi_ref(o_t, l) = sum_b p(V = b) v(b),
            A^pi_ref(o_t, a_t) (Appendix F), eps_l, I_t (Eq. 3), Eq. 13
Retrieved:  2026-10-12
License:    n/a: transcribed from formulas. No text or code is copied; the paper
            itself is distributed under the arXiv non-exclusive distribution
            license 1.0.

WHY THIS FILE EXISTS
--------------------
Physical Intelligence released no RECAP code (openpi has pi0 and pi0.5 only),
so recap_advantage_conditioning has no upstream implementation to vendor. This
module writes the paper's definitions down as plain Python loops over lists of
floats, with no torch, so it cannot share a vectorization or indexing bug with
the torch reference solution.

The transcribed definitions, in plain text:

  r_t = 0        if t = T and success                       (Eq. 5)
      = -C_fail  if t = T and failure
      = -1       otherwise
  values are normalized per task by the maximum episode length    (Sec. V-C)
  V(o_t, l) = sum over b of p_phi(V = b | o_t) v(b)               (Sec. IV-A)
  A(o_t, a_t) = sum over t' = t .. t+N-1 of r_t' + V(o_{t+N}) - V(o_t)
                N = 50 in post-training; N = T in pre-training      (App. F)
  I_t = 1[A(o_t, a_t, l) > eps_l]                                  (Eq. 3)
  I_t = True for actions provided as human corrections            (Sec. IV-B)
  eps_l: "such that approximately 30% of the demonstration data has
         positive advantage" (pre-training); about 40% of evaluation rollouts
         in fine-tuning; about 10% for the T-shirt and shorts task  (App. F)
  drop the indicator 30% of the time                              (App. F)
  guided direction = grad log pi(a|o, l)
                     + beta (grad log pi(a|I, o, l) - grad log pi(a|o, l))  (Eq. 13)

WHAT WAS CHANGED, AND WHY
-------------------------
  * Threshold ambiguity. Section V-D says "We set eps_l to the 30% percentile
    of values predicted by the value function for the task l", while Appendix F
    says the threshold is chosen "such that approximately 30% of the
    demonstration data has positive advantage". Read literally, the first puts
    eps_l at the 30th percentile, so about 70% would be positive. This module
    follows Appendix F, the more specific statement: eps_l is the
    (1 - positive_fraction) quantile of the task's advantages, so about
    positive_fraction of them satisfy A > eps_l. The paper says
    "approximately" and does not name a quantile rule; this module uses linear
    interpolation between order statistics (the common "type 7" rule), written
    out by hand.
  * Pre-training advantage. Appendix F writes the N = T estimate as
    sum over t' = 0 .. T of r_t' - V(o_t), "setting N = T for each episode".
    Substituting N = T into the n-step formula gives a sum that starts at t,
    not 0; a sum from 0 would add the rewards already received before t. This
    module reads the lower limit as t: with no lookahead, A_t = R_t - V(o_t).
  * Episode boundary. The paper does not say what V(o_{t+N}) is when t + N runs
    past the last observation. Here the sum stops at the last reward and the
    bootstrap term is 0 (no observation exists after the terminal one). When
    t + N lands exactly on the last observation, its predicted value is used.
  * Indexing. An episode of length T has observations o_0 .. o_T and rewards
    r_0 .. r_T, with the terminal reward r_T, as in Eq. 5. Rewards are divided
    by the task's maximum episode length so returns live on the same scale as
    the normalized values. The paper does not clip failed returns into
    (-1, 0); discretization below clamps them to the end bins instead.
  * Discretization. Eq. 1 uses B = 201 bins but the paper does not give the bin
    rule. This module places B evenly spaced bin values on [-1, 0] and maps a
    return to the nearest one, clamped.
  * Threshold population and corrections. Appendix F sets the threshold from
    per-iteration evaluation rollouts in fine-tuning and a random 10k
    demonstration sample in pre-training; pooling every sample of task l, as
    here, is a simplification. The paper does not say whether corrections
    enter the threshold computation. Here eps_l is computed over
    every sample of task l, and corrections are forced positive afterwards.
  * Dropout. Appendix F drops the indicator 30% of the time; here a caller
    supplies a uniform draw u per sample and the indicator is dropped when
    u < drop_prob, so the test is deterministic.

NOTHING HERE IMPORTS A TASK SOLUTION. This module must stay independent of the
answers it is used to check.
"""

from __future__ import annotations

import math

__all__ = [
    "bin_values",
    "cfg_direction",
    "discretize_return",
    "eq5_rewards",
    "expected_value",
    "improvement_codes",
    "nstep_advantages",
    "returns_to_go",
    "threshold_for_fraction",
]

POSITIVE, NEGATIVE, DROPPED = 1, 0, -1
INDICATOR_TEXT = {POSITIVE: "Advantage: positive", NEGATIVE: "Advantage: negative", DROPPED: ""}


def eq5_rewards(T: int, success: bool, max_len: float, c_fail: float) -> list[float]:
    """Eq. 5 for t = 0 .. T, each divided by the task's maximum episode length (Sec. V-C)."""
    out = []
    for t in range(T + 1):
        if t == T:
            r = 0.0 if success else -c_fail
        else:
            r = -1.0
        out.append(r / max_len)
    return out


def returns_to_go(rewards: list[float]) -> list[float]:
    """R_t = sum over t' = t .. T of r_t' (Sec. IV-A, no discount)."""
    out = []
    for t in range(len(rewards)):
        total = 0.0
        for k in range(t, len(rewards)):
            total += rewards[k]
        out.append(total)
    return out


def bin_values(B: int) -> list[float]:
    """v(b) for b = 0 .. B-1, evenly spaced over the normalized value range [-1, 0]."""
    return [-1.0 + b / (B - 1) for b in range(B)]


def discretize_return(R: float, B: int) -> int:
    """The bin whose value v(b) is nearest to R, clamped to 0 .. B-1 (the target in Eq. 1)."""
    values = bin_values(B)
    best, best_gap = 0, math.inf
    for b, v in enumerate(values):
        gap = abs(R - v)
        if gap < best_gap:
            best, best_gap = b, gap
    return best


def expected_value(logits: list[float]) -> float:
    """V = sum over b of p(V = b) v(b) with p a softmax over the B logits (Sec. IV-A)."""
    top = max(logits)
    weights = [math.exp(x - top) for x in logits]
    z = sum(weights)
    values = bin_values(len(logits))
    return sum(w / z * v for w, v in zip(weights, values))


def nstep_advantages(rewards: list[float], values: list[float], N: int | None) -> list[float]:
    """Appendix F: sum of the next N rewards plus V(o_{t+N}) minus V(o_t); N = None means N = T."""
    last = len(rewards) - 1
    out = []
    for t in range(len(rewards)):
        total = 0.0
        k = t
        while k <= last and (N is None or k < t + N):
            total += rewards[k]
            k += 1
        bootstrap = 0.0
        if N is not None and t + N <= last:
            bootstrap = values[t + N]
        out.append(total + bootstrap - values[t])
    return out


def threshold_for_fraction(advantages: list[float], positive_fraction: float) -> float:
    """eps_l with about positive_fraction of advantages above it (Appendix F), type-7 quantile."""
    ordered = sorted(advantages)
    position = (len(ordered) - 1) * (1.0 - positive_fraction)
    lo = math.floor(position)
    hi = min(lo + 1, len(ordered) - 1)
    frac = position - lo
    return ordered[lo] + (ordered[hi] - ordered[lo]) * frac


def improvement_codes(
    advantages: list[float],
    tasks: list[int],
    positive_fraction: float,
    corrections: list[bool],
    u: list[float],
    drop_prob: float,
) -> list[int]:
    """I_t per sample: per-task eps_l (App. F), A > eps_l (Eq. 3), corrections True (IV-B), dropout (App. F)."""
    eps = {}
    for task in set(tasks):
        eps[task] = threshold_for_fraction(
            [a for a, k in zip(advantages, tasks) if k == task], positive_fraction
        )
    codes = []
    for a, k, fixed, draw in zip(advantages, tasks, corrections, u):
        if draw < drop_prob:
            codes.append(DROPPED)
        elif fixed or a > eps[k]:
            codes.append(POSITIVE)
        else:
            codes.append(NEGATIVE)
    return codes


def cfg_direction(uncond: list[float], cond: list[float], beta: float) -> list[float]:
    """Eq. 13, applied elementwise to the unconditional and indicator-conditioned outputs."""
    return [g_u + beta * (g_c - g_u) for g_u, g_c in zip(uncond, cond)]
