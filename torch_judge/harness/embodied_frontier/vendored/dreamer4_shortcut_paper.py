"""Dreamer 4 shortcut forcing objective, transcribed from the paper's equations.

Upstream:   paper only (no official code). Hafner, Yan, Lillicrap, "Training
            Agents Inside of Scalable World Models", arXiv 2509.24527.
            https://arxiv.org/abs/2509.24527
Commit:     arXiv 2509.24527v1 (LaTeX source dated 30 Sep 2025)
Files:      Section 2 "Background", paragraph "Shortcut models" (Equations 3
            and 4); Section 3.2 "Interactive Dynamics", paragraph "Shortcut
            forcing" (Equations 6, 7 and 8 and footnote 1 on the x-to-v
            conversion); Section 4.4 "Model Design" (the cascade table).
Symbols:    p(tau, d) (Equation 4), z~ (Equation 6), b', z', b'' and the
            two-case loss (Equation 7), w(tau) (Equation 8)
Retrieved:  2026-10-12
License:    n/a: transcribed from formulas, no code copied; the paper is
            distributed under the arXiv non-exclusive distribution license 1.0.
            ``shortcut_models_bootstrap_target`` alone follows
            kvfrans/shortcut-models targets_shortcut.py at commit
            601004348667094e1b71f30942199759412d4432 (MIT License,
            Copyright (c) 2024 Kevin Frans).

WHY THIS FILE EXISTS
--------------------
Dreamer 4 released no code, so shortcut_forcing_loss has no upstream
implementation to vendor. This module writes the paper's equations down one
frame and one feature at a time, with Python floats and plain loops, so it
cannot share a vectorization or broadcasting bug with the torch reference.
The model is still called on whole (B, T, D) tensors, because Equation 7
applies f_theta to the whole corrupted sequence; only the arithmetic around
the calls is scalar.

The transcribed equations, in plain text (tau = signal level, 0 is pure noise
and 1 is clean data; d = step size; sg = stop gradient):

  d ~ 1 / U({1, 2, 4, 8, ..., K_max})
  tau ~ U({0, 1/d, ..., 1 - 1/d})                                 (Equation 4)
  z~ = (1 - tau) z_0 + tau z_1,   z1_hat = f(z~, tau, d, a)        (Equation 6)
  b'  = (f(z~, tau, d/2, a) - z_tau) / (1 - tau)
  z'  = z~ + b' d/2
  b'' = (f(z', tau + d/2, d/2, a) - z') / (1 - (tau + d/2))
  L = ||z1_hat - z_1||^2                                    if d = d_min
  L = (1 - tau)^2 || (z1_hat - z~)/(1 - tau) - sg(b_1 + b_2)/2 ||^2  else
                                                                   (Equation 7)
  w(tau) = 0.9 tau + 0.1                                           (Equation 8)

WHAT WAS CHANGED, AND WHY
-------------------------
  * Equation 4 prints the level grid as {0, 1/d, ..., 1 - 1/d}. With d <= 1
    that set lies outside [0, 1] whenever d < 1, so it is read as
    {0, d, 2d, ..., 1 - d}: "the grid that is reached by the current step
    size" (the sentence before Equation 4), which is also what Frans et al.'s
    code samples (t = randint(0, 2**dt_base) / 2**dt_base).
  * Equation 7 writes z_tau in b' and sg(b_1 + b_2) in the target. z_tau is
    the corrupted input z~ of Equation 6, and b_1, b_2 are b' and b''. Both
    are notation slips; this module uses z~, b' and b''.
  * The paper never writes w(tau) times L. Equation 8 introduces "a ramp loss
    weight" as a function of tau alone, the paragraph before it motivates it
    with both the flow term and the bootstrap term at low signal levels, and
    Section 4.4 applies the ramp weight to a pure diffusion forcing baseline
    (flow term only). This module therefore multiplies BOTH branches of
    Equation 7 by w(tau). That reading is the exercise's contract.
  * ||.||_2^2 is the sum of squares over the feature axis, giving one loss per
    (batch, frame). The paper's further normalization of each loss term by a
    running RMS estimate (Section 3, opening paragraph) and the reduction to
    a scalar are left to the caller.
  * The action input a is dropped; it is passed through unchanged in every
    call and does not touch the arithmetic.
  * The two half-step calls get the whole sequence, with d/2 and tau + d/2 at
    every frame, as Equation 7 is written in vector form. Frames with
    d = d_min take the flow branch and their half-step outputs are unused.
  * The bootstrap target is computed under torch.no_grad(); that is sg.

NOTHING HERE IMPORTS A TASK SOLUTION. This module must stay independent of the
answers it is used to check.
"""

from __future__ import annotations

import torch

__all__ = [
    "frame_losses",
    "ramp_weight",
    "shortcut_models_bootstrap_target",
    "step_level_support",
]


def step_level_support(k_max: int) -> dict[tuple[float, float], float]:
    """Equation 4: every (d, tau) pair and its probability, with the grid read as {0, d, ..., 1 - d}."""
    counts = []
    k = 1
    while k <= k_max:
        counts.append(k)
        k *= 2
    assert counts[-1] == k_max, "K_max must be a power of two"
    support: dict[tuple[float, float], float] = {}
    for k in counts:
        d = 1.0 / k
        for i in range(k):
            support[(d, i * d)] = (1.0 / len(counts)) * (1.0 / k)
    return support


def ramp_weight(tau: float) -> float:
    """Equation 8."""
    return 0.9 * tau + 0.1


def _sq_norm(values: list[float]) -> float:
    total = 0.0
    for v in values:
        total += v * v
    return total


def frame_losses(model, z1: torch.Tensor, z0: torch.Tensor, tau: torch.Tensor, d: torch.Tensor,
                 d_min: float) -> list[list[float]]:
    """Equations 6-8: the ramp-weighted loss of every (batch, frame), as nested Python lists."""
    B, T, D = z1.shape
    z_tilde = torch.empty_like(z1)
    for b in range(B):
        for t in range(T):
            s = float(tau[b, t])
            for k in range(D):
                z_tilde[b, t, k] = (1 - s) * float(z0[b, t, k]) + s * float(z1[b, t, k])

    with torch.no_grad():
        z1_hat = model(z_tilde.clone(), tau.clone(), d.clone())
        half = d / 2
        first = model(z_tilde.clone(), tau.clone(), half.clone())
        b_prime = [[[0.0] * D for _ in range(T)] for _ in range(B)]
        z_prime = torch.empty_like(z1)
        for b in range(B):
            for t in range(T):
                s, h = float(tau[b, t]), float(d[b, t]) / 2
                for k in range(D):
                    b_prime[b][t][k] = (float(first[b, t, k]) - float(z_tilde[b, t, k])) / (1 - s)
                    z_prime[b, t, k] = float(z_tilde[b, t, k]) + b_prime[b][t][k] * h
        second = model(z_prime.clone(), (tau + half).clone(), half.clone())

    out = []
    for b in range(B):
        row = []
        for t in range(T):
            s, step = float(tau[b, t]), float(d[b, t])
            if step == d_min:
                loss = _sq_norm([float(z1_hat[b, t, k]) - float(z1[b, t, k]) for k in range(D)])
            else:
                h = step / 2
                residual = []
                for k in range(D):
                    b_second = (float(second[b, t, k]) - float(z_prime[b, t, k])) / (1 - (s + h))
                    target = (b_prime[b][t][k] + b_second) / 2
                    v_hat = (float(z1_hat[b, t, k]) - float(z_tilde[b, t, k])) / (1 - s)
                    residual.append(v_hat - target)
                loss = (1 - s) ** 2 * _sq_norm(residual)
            row.append(ramp_weight(s) * loss)
        out.append(row)
    return out


def shortcut_models_bootstrap_target(v_model, x_t: torch.Tensor, t: torch.Tensor, dt: torch.Tensor,
                                     clip: float | None = 4.0) -> torch.Tensor:
    """Frans et al.'s v-space bootstrap target (targets_shortcut.py, get_targets, step 3, no CFG).

    Transcribed from JAX. Upstream passes dt_base_bootstrap (log2 of 1/dt, plus
    one) to the network and reads the EMA parameters; here v_model takes the
    half step size dt / 2 itself and the caller chooses the parameters.
    ``clip=None`` drops upstream's jnp.clip(..., -4, 4) on x_t2 and v_target.
    """
    dt_bootstrap = dt / 2
    v_b1 = v_model(x_t, t, dt_bootstrap)
    t2 = t + dt_bootstrap
    x_t2 = x_t + dt_bootstrap[..., None] * v_b1
    if clip is not None:
        x_t2 = torch.clamp(x_t2, -clip, clip)
    v_b2 = v_model(x_t2, t2, dt_bootstrap)
    v_target = (v_b1 + v_b2) / 2
    if clip is not None:
        v_target = torch.clamp(v_target, -clip, clip)
    return v_target
