"""Mutation gates and metadata contracts for the parameter-efficient fine-tuning path."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from torch_judge.tasks import get_task
from torch_judge.tasks._schema import validate_task
from tests.quality.mutation_runner import Mutation, assert_mutations_rejected
from tests.quality.test_harness_paths_mutations import _seeded_task_variant

ROOT = Path(__file__).resolve().parents[2]
PATH_ID = "peft"

NEW_TASKS = ["lora_merge", "dora", "prefix_tuning"]

# (name, text in the reference solution, replacement). Each replacement is one realistic bug.
MUTATIONS = {
    "lora_merge": [
        ("forgets the scaling", "return self.lora_B @ self.lora_A * self.scaling", "return self.lora_B @ self.lora_A"),
        ("merged forward adds the adapter again", "        if self.merged:\n            return self.linear(x)\n", ""),
        ("merges twice", "        if self.merged:\n            return\n        self.linear.weight.add_", "        self.linear.weight.add_"),
        ("unmerges without a merge", "        if not self.merged:\n            return\n", ""),
        ("unmerge adds", "self.linear.weight.sub_(self.delta_weight())", "self.linear.weight.add_(self.delta_weight())"),
        ("replaces the Parameter", "self.linear.weight.add_(self.delta_weight())",
         "self.linear.weight = nn.Parameter(self.linear.weight + self.delta_weight(), requires_grad=False)"),
        ("records autograd history", "    @torch.no_grad()\n    def merge(self):\n        if self.merged:\n            return\n        self.linear.weight.add_(self.delta_weight())",
         "    def merge(self):\n        if self.merged:\n            return\n        self.linear.weight = self.linear.weight + self.delta_weight()"),
        ("scaling uses rank over alpha", "self.scaling = alpha / rank", "self.scaling = rank / alpha"),
    ],
    "dora": [
        ("norm not detached", "norm = weight.norm(dim=1).detach()", "norm = weight.norm(dim=1)"),
        ("norm over input columns", "norm = weight.norm(dim=1).detach()", "norm = weight.norm(dim=0).detach().mean()"),
        ("norm of the base weight only", "norm = weight.norm(dim=1).detach()", "norm = self.linear.weight.norm(dim=1).detach()"),
        ("norm skips the scaling", "norm = weight.norm(dim=1).detach()",
         "norm = (self.linear.weight + self.lora_B @ self.lora_A).norm(dim=1).detach()"),
        ("rescales the bias", "return F.linear(x, weight, self.linear.bias)",
         "return F.linear(x, weight, self.linear.bias * self.magnitude / norm)"),
        ("magnitude starts at one", "self.magnitude = nn.Parameter(self.linear.weight.detach().norm(dim=1))",
         "self.magnitude = nn.Parameter(torch.ones(out_features))"),
        ("magnitude frozen", "self.magnitude = nn.Parameter(self.linear.weight.detach().norm(dim=1))",
         "self.magnitude = nn.Parameter(self.linear.weight.detach().norm(dim=1), requires_grad=False)"),
        ("plain LoRA", "        weight = (self.magnitude / norm)[:, None] * weight\n", ""),
    ],
    "prefix_tuning": [
        ("causal mask over the whole key axis", "        allowed = torch.cat([prefix_ok, token_ok], dim=2)[:, None]\n",
         "        allowed = torch.cat([prefix_ok, token_ok], dim=2)[:, None]\n        allowed = allowed & torch.ones(T, P + T, dtype=torch.bool, device=x.device).tril()\n"),
        ("padding mask shifted onto the prefix", "token_ok = causal[None, :, :] & attention_mask.bool()[:, None, :]",
         "token_ok = causal[None, :, :] & torch.cat([torch.ones(B, P, dtype=torch.bool), attention_mask.bool()], dim=1)[:, None, :T]"),
        ("ignores padding", "token_ok = causal[None, :, :] & attention_mask.bool()[:, None, :]",
         "token_ok = causal[None, :, :].expand(B, T, T)"),
        ("no causal mask", "causal = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()",
         "causal = torch.ones(T, T, dtype=torch.bool, device=x.device)"),
        ("prefix through k_proj", "k = torch.cat([self.prefix_k.expand(B, H, P, hd), k], dim=2)",
         "k = torch.cat([self.k_proj(self.prefix_k.transpose(0, 1).reshape(P, D)).view(P, H, hd).transpose(0, 1).expand(B, H, P, hd), k], dim=2)"),
        ("prefix keys reused as values", "v = torch.cat([self.prefix_v.expand(B, H, P, hd), v], dim=2)",
         "v = torch.cat([self.prefix_k.expand(B, H, P, hd), v], dim=2)"),
        ("scales by d_model", "/ math.sqrt(hd)", "/ math.sqrt(D)"),
        ("projections trainable", "            proj.requires_grad_(False)\n", "            pass\n"),
        ("accepts an empty prefix", "if d_model % num_heads != 0 or num_prefix < 1:", "if d_model % num_heads != 0:"),
    ],
}


@pytest.mark.parametrize("repeat", range(3))
@pytest.mark.parametrize("task_id,targets", MUTATIONS.items())
def test_mutations_rejected_across_distinct_seeds(task_id, targets, repeat):
    original = get_task(task_id)["solution"]
    mutations = []
    for name, old, new in targets:
        assert old in original, f"mutation target drifted: {task_id}/{name}"
        mutations.append(Mutation(name, original.replace(old, new, 1)))
    rejected = assert_mutations_rejected(
        task_id, mutations, require_unshown=True,
        task_override=_seeded_task_variant(task_id, repeat),
    )
    assert set(rejected) == {name for name, _, _ in targets}


@pytest.mark.parametrize("task_id", NEW_TASKS)
def test_task_metadata_is_valid_and_cited(task_id):
    task = get_task(task_id)
    validate_task(task_id, task)
    assert not {"title_zh", "description_zh", "hint_zh"} & set(task)
    assert task["interview_questions"]
    assert any(source["kind"] == "code" for source in task["sources"])


def test_every_task_has_mutations():
    assert set(MUTATIONS) == set(NEW_TASKS)


def test_peft_path_is_ordered_and_listed():
    paths = {p["id"]: p for p in json.loads((ROOT / "web/src/lib/paths.json").read_text())["paths"]}
    assert paths[PATH_ID]["problems"] == ["lora", "lora_merge", "dora", "qlora", "prefix_tuning"]
    # LoRA and QLoRA also stay in the efficiency path they came from.
    assert {"lora", "qlora"} <= set(paths["training-inference-efficiency"]["problems"])
    starters = json.loads((ROOT / "web/src/lib/starters.json").read_text())
    assert all(task_id in starters for task_id in NEW_TASKS)
