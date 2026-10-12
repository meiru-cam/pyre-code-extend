"""Mutation gate and metadata contract for the latent_action_quantizer exercise."""

from __future__ import annotations

import pytest

from torch_judge.tasks import get_task
from torch_judge.tasks._schema import validate_task
from tests.quality.mutation_runner import Mutation, assert_mutations_rejected

TASK_ID = "latent_action_quantizer"
REFERENCE = get_task(TASK_ID)["solution"]


def edit(*replacements: tuple[str, str]) -> str:
    """The reference with exact-substring edits; fails loudly if a target drifts."""
    code = REFERENCE
    for old, new in replacements:
        assert code.count(old) == 1, f"mutation target not unique or missing: {old!r}"
        code = code.replace(old, new)
    return code


STE = "z_q = x + (q - x).detach()"
NSVQ = "z_q = x + (residual_norm / noise_norm + eps) * noise"
RESIDUAL = "residual_norm = (x - q).pow(2).sum(-1, keepdim=True).sqrt()"
ARGMIN = "idx_flat = torch.argmin(dist, dim=-1)"
PERPLEXITY = "perplexity = torch.exp(-(probs * torch.log(probs + eps)).sum())"
DONORS = "donors = live[torch.arange(dead.numel()) % live.numel()]"

LAQ_MUTATIONS = [
    # Research sketch: the frame difference, STE gradients, NSVQ, perplexity and argmin ties.
    Mutation("reversed_difference", edit(("x = f_tH - f_t", "x = f_t - f_tH"))),
    Mutation("ste_returns_hard_code", edit((STE, "z_q = q"))),
    Mutation("ste_detaches_the_encoder_side", edit((STE, "z_q = x + (q.detach() - x)"))),
    Mutation("nsvq_eps_in_denominator", edit((NSVQ, "z_q = x + (residual_norm / (noise_norm + eps)) * noise"))),
    Mutation("nsvq_detached_residual_norm", edit((RESIDUAL, RESIDUAL + ".detach()"))),
    Mutation("perplexity_without_eps", edit((PERPLEXITY, "perplexity = torch.exp(-(probs * torch.log(probs)).sum())"))),
    Mutation("ties_pick_highest_index", edit((
        ARGMIN, "idx_flat = (dist.shape[-1] - 1) - torch.argmin(dist.flip(-1), dim=-1)",
    ))),
    # Further plausible slips.
    Mutation("l1_distance", edit((
        """dist = (flat.pow(2).sum(-1, keepdim=True)
            - 2 * flat @ codebook.t()
            + codebook.pow(2).sum(-1)[None, :])""",
        "dist = (flat[:, None, :] - codebook[None, :, :]).abs().sum(-1)",
    ))),
    Mutation("beta_on_codebook_term", edit(("vq_loss = codebook_loss + beta * commit_loss", "vq_loss = beta * codebook_loss + commit_loss"))),
    Mutation("vq_loss_without_stop_gradients", edit(
        ("codebook_loss = (x.detach() - q).pow(2).mean()", "codebook_loss = (x - q).pow(2).mean()"),
        ("commit_loss = (x - q.detach()).pow(2).mean()", "commit_loss = (x - q).pow(2).mean()"),
    )),
    Mutation("perplexity_is_entropy", edit((PERPLEXITY, "perplexity = -(probs * torch.log(probs + eps)).sum()"))),
    Mutation("perplexity_over_first_row_only", edit((
        "counts = torch.bincount(idx_flat, minlength=K).to(x.dtype)",
        "counts = torch.bincount(idx.reshape(-1, idx.shape[-1])[0] if idx.ndim > 1 else idx_flat, minlength=K).to(x.dtype)",
    ))),
    Mutation("nsvq_missing_noise_not_rejected", edit((
        "if noise is None or noise.shape != x.shape:", "if noise is None and False:",
    ))),
    Mutation("dead_at_threshold", edit((
        "dead = torch.nonzero(rate < threshold).flatten()", "dead = torch.nonzero(rate <= threshold).flatten()",
    ), (
        "live = torch.nonzero(rate >= threshold).flatten()", "live = torch.nonzero(rate > threshold).flatten()",
    ))),
    Mutation("every_dead_code_copies_first_live", edit((DONORS, "donors = live[torch.zeros(dead.numel(), dtype=torch.long)]"))),
    Mutation("jitter_on_every_row", edit((
        "new[dead] = new[dead] + jitter[dead]", "new = new + jitter",
    ))),
    Mutation("replaces_in_place", edit((
        "new = source.clone()", "new = source",
    ))),
    Mutation("no_live_copies_from_dead", edit((
        """        if live.numel() == 0:
            if jitter is not None:
                new = new + jitter
            return new, dead
""", """        if live.numel() == 0:
            return new, dead
""",
    ))),
]


def test_latent_action_quantizer_metadata_is_valid():
    task = get_task(TASK_ID)
    validate_task(TASK_ID, task, known_ids=None)
    assert not any(key.endswith("_zh") for key in task), "new exercises are English only"
    for code in [task["solution"]] + [test["code"] for test in task["tests"]]:
        assert "scipy" not in code and "gymnasium" not in code


def test_latent_action_quantizer_code_sources_are_pinned():
    pins = {
        "laq/laq_model/nsvq.py": ("46aca51d7faebcec02d7d323bbb3820c2df07bc6", "MIT"),
        "latent_action_model/genie/modules/blocks.py": ("0ab9e9dd3074ffa14fdd1a00aa2aa84e4f8e1352", "Apache-2.0"),
        "latent_action_model/genie/model.py": ("0ab9e9dd3074ffa14fdd1a00aa2aa84e4f8e1352", "Apache-2.0"),
    }
    code = [source for source in get_task(TASK_ID)["sources"] if source["kind"] == "code"]
    assert {source["path"] for source in code} == set(pins)
    for source in code:
        commit, license_ = pins[source["path"]]
        assert source["commit"] == commit and commit in source["url"]
        assert source["license"] == license_


@pytest.mark.parametrize("_repeat", range(3))
def test_latent_action_quantizer_reference_and_mutations(_repeat):
    assert set(
        assert_mutations_rejected(TASK_ID, LAQ_MUTATIONS, require_unshown=True)
    ) == {m.name for m in LAQ_MUTATIONS}
