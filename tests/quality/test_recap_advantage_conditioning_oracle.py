"""recap_advantage_conditioning's reference solution, checked against the paper.

No official RECAP code exists, so the oracle is the pi*0.6 paper's own
definitions (arXiv 2511.14759v2: Eq. 1, 3, 5 and 13, Sections IV-A, IV-B, V-B to
V-D and Appendix F), transcribed in plain Python loops in
``torch_judge.harness.embodied_frontier.vendored.recap_paper``. The two pinned
divergences from a literal reading of the paper (threshold per Appendix F, and
the N = T sum starting at t) are asserted at the bottom.
"""

from __future__ import annotations

import pytest
import torch

from torch_judge.harness.embodied_frontier.vendored import recap_paper as paper
from torch_judge.tasks import get_task

TASK_ID = "recap_advantage_conditioning"


def reference(num_bins: int = 201):
    task = get_task(TASK_ID)
    assert task is not None, f"unknown task {TASK_ID!r}"
    namespace: dict = {}
    exec(task["solution"], namespace)
    return namespace[task["function_name"]](num_bins)


def close(a, b, tol=1e-10):
    return len(a) == len(b) and all(abs(x - y) <= tol for x, y in zip(a, b))


@pytest.mark.parametrize("seed", range(8))
def test_rewards_returns_and_advantages_match_paper(seed):
    rc = reference()
    g = torch.Generator().manual_seed(seed)
    T = int(torch.randint(0, 60, (), generator=g))
    max_len = T + int(torch.randint(1, 30, (), generator=g))
    c_fail = float(torch.rand((), generator=g) * 100)
    success = bool(seed % 2)
    rewards = rc.episode_rewards(T, success, max_len, c_fail)
    want = paper.eq5_rewards(T, success, max_len, c_fail)
    assert close(rewards.tolist(), want)
    assert close(rc.episode_returns(T, success, max_len, c_fail).tolist(), paper.returns_to_go(want))
    values = (-torch.rand(T + 1, generator=g, dtype=torch.float64)).tolist()
    for n in (1, 3, 50, T, T + 1, None):
        if n == 0:  # T can be 0, and a lookahead must be positive
            continue
        got = rc.nstep_advantage(rewards, torch.tensor(values, dtype=torch.float64), n).tolist()
        assert close(got, paper.nstep_advantages(want, values, n)), (seed, n)


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("num_bins", [201, 21])
def test_value_bins_and_expected_value_match_paper(seed, num_bins):
    rc = reference(num_bins)
    g = torch.Generator().manual_seed(100 + seed)
    R = torch.rand(40, generator=g, dtype=torch.float64) * 1.6 - 1.3
    width = 1.0 / (num_bins - 1)
    frac = ((R + 1) / width) % 1.0
    R = torch.where((frac - 0.5).abs() < 1e-3, R + 0.1 * width, R)  # keep away from exact ties
    assert rc.value_bins(R).tolist() == [paper.discretize_return(float(r), num_bins) for r in R]
    logits = torch.randn(5, num_bins, generator=g, dtype=torch.float64) * 4
    got = rc.expected_value(logits).tolist()
    assert close(got, [paper.expected_value(row.tolist()) for row in logits])


@pytest.mark.parametrize("seed", range(10))
def test_indicators_match_paper(seed):
    rc = reference()
    g = torch.Generator().manual_seed(200 + seed)
    m = int(torch.randint(5, 120, (), generator=g))
    adv = torch.randn(m, generator=g, dtype=torch.float64)
    if seed % 3 == 0:
        adv = torch.round(adv * 2) / 2  # many ties at the threshold
    tasks = torch.randint(0, 3, (m,), generator=g)
    corr = torch.rand(m, generator=g) < 0.15
    u = torch.rand(m, generator=g, dtype=torch.float64)
    q = [0.3, 0.4, 0.1][seed % 3]
    p = [0.3, 0.0, 0.5][seed % 3]
    got = rc.indicators(adv, tasks, q, corr, u, p).tolist()
    want = paper.improvement_codes(adv.tolist(), tasks.tolist(), q, corr.tolist(), u.tolist(), p)
    assert got == want, seed
    assert [rc.indicator_text(c) for c in got] == [paper.INDICATOR_TEXT[c] for c in want]
    for task in set(tasks.tolist()):
        rows = [a for a, k in zip(adv.tolist(), tasks.tolist()) if k == task]
        assert abs(rc.threshold(torch.tensor(rows, dtype=torch.float64), q) - paper.threshold_for_fraction(rows, q)) < 1e-12


@pytest.mark.parametrize("beta", [0.0, 1.0, 1.5, 2.5])
def test_cfg_matches_eq13(beta):
    rc = reference()
    g = torch.Generator().manual_seed(int(beta * 10))
    vu = torch.randn(12, generator=g, dtype=torch.float64)
    vc = torch.randn(12, generator=g, dtype=torch.float64)
    assert close(rc.cfg_velocity(vu, vc, beta).tolist(), paper.cfg_direction(vu.tolist(), vc.tolist(), beta))


@pytest.mark.parametrize(
    "rewards,values,n",
    [
        ([0.0], [0.3], None),        # a single observation
        ([0.0], [0.3], 1),
        ([-1.0, -1.0, 0.0], [-2.0, -1.0, 0.0], 2),   # t + N lands on the last observation
        ([-1.0, -1.0, -9.0], [-5.0, -4.0, -3.0], 5),  # N past the end
    ],
)
def test_edge_episodes(rewards, values, n):
    rc = reference()
    got = rc.nstep_advantage(torch.tensor(rewards, dtype=torch.float64), torch.tensor(values, dtype=torch.float64), n)
    assert close(got.tolist(), paper.nstep_advantages(rewards, values, n))


def test_pinned_reading_threshold_follows_appendix_f():
    # Section V-D, read literally, would put eps at the 30th percentile (about 70% positive).
    # Appendix F says about 30% positive; the exercise and the oracle follow Appendix F.
    rc = reference()
    adv = torch.arange(100, dtype=torch.float64)
    codes = rc.indicators(adv, torch.zeros(100, dtype=torch.long), 0.3, torch.zeros(100, dtype=torch.bool), torch.ones(100))
    assert int((codes == 1).sum()) == 30
    assert rc.threshold(adv, 0.3) == pytest.approx(69.3)


def test_pinned_reading_pretraining_sum_starts_at_t():
    # Appendix F writes the N = T estimate as a sum from t' = 0; the exercise starts it at t.
    rc = reference()
    rewards = rc.episode_rewards(3, True, 3, 1.0)
    values = torch.tensor([-1.0, -2.0 / 3, -1.0 / 3, 0.0], dtype=torch.float64)  # the exact value
    assert torch.allclose(rc.nstep_advantage(rewards, values), torch.zeros(4, dtype=torch.float64), atol=1e-12)
