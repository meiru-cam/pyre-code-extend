"""Registration and grading gate for data diagnostics and distillation exercises."""

import pytest

from grading_service.main import _execute_tests
from torch_judge.tasks import get_task


NEW_TASKS = (
    "rl_data_audit",
    "masked_perplexity",
    "confidence_calibration",
    "embedding_diversity_selection",
    "gradient_orthogonality_audit",
    "masked_kd_kl",
    "opd_sampled_kl_advantage",
    "opsd_privileged_context_kl",
)

CODE_ORIGINS = {
    "rl_data_audit": ("src/datasets/dataset_dict.py", "DatasetDict.unique"),
    "masked_perplexity": ("nano_r1_script.py", "compute_pg_loss"),
    "confidence_calibration": ("src/torchmetrics/functional/classification/calibration_error.py", "_binning_bucketize; _ce_compute"),
    "embedding_diversity_selection": ("sklearn/metrics/pairwise.py", "cosine_similarity"),
    "gradient_orthogonality_audit": ("sklearn/metrics/pairwise.py", "cosine_similarity"),
    "masked_kd_kl": ("trl/experimental/gkd/gkd_trainer.py", "GKDTrainer.generalized_jsd_loss"),
    "opd_sampled_kl_advantage": ("slime/backends/megatron_utils/loss.py", "apply_opd_kl_to_advantages"),
    "opsd_privileged_context_kl": ("opsd_trainer.py", "OPSDTrainer.generalized_jsd_loss"),
}


@pytest.mark.parametrize("task_id", NEW_TASKS)
def test_code_provenance_matches_relevant_implementation(task_id):
    code = [source for source in get_task(task_id)["sources"] if source["kind"] == "code"]
    assert any((source["path"], source["symbol"]) == CODE_ORIGINS[task_id] for source in code)


@pytest.mark.parametrize("task_id", NEW_TASKS)
def test_reference_passes_task_contract(task_id):
    task = get_task(task_id)
    assert task is not None, f"{task_id} is not registered"
    result = _execute_tests(task["solution"], task, capture_output=False)
    assert result.allPassed, [(r.name, r.error) for r in result.results if not r.passed]
