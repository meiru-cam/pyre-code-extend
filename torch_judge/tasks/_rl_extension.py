"""Small metadata helpers for the eight data/distillation RL exercises."""

NANO = {
    "kind": "code", "url": "https://github.com/McGill-NLP/nano-aha-moment",
    "commit": "5314e6f8fc60efaa0f4b8fdb62353e9bd451638a",
    "path": "nano_r1_script.py", "symbol": "compute_pg_loss", "license": "MIT",
    "adapted": "Per-token policy log probabilities and masked RL objectives.",
    "simplifications": "Pure CPU tensors; no model forward pass or distributed rollout.",
}
DATASETS = {
    "kind": "code", "url": "https://github.com/huggingface/datasets",
    "commit": "a4ee9cf1b9ba07d59766b17c33f91412a69969c4",
    "path": "src/datasets/dataset_dict.py", "symbol": "DatasetDict.unique",
    "license": "Apache-2.0", "adapted": "Enumerates unique values of a column separately for each split.",
    "simplifications": "Counts exact sample IDs in plain records to also detect within-split duplicates and cross-split overlap; no Arrow dataset.",
}
TORCHMETRICS = {
    "kind": "code", "url": "https://github.com/Lightning-AI/torchmetrics",
    "commit": "6c551c6469dc04fe92291c000ce2607860df92d2",
    "path": "src/torchmetrics/functional/classification/calibration_error.py",
    "symbol": "_binning_bucketize; _ce_compute", "license": "Apache-2.0",
    "adapted": "Confidence binning and expected calibration error.",
    "simplifications": "Binary correctness labels and scalar CPU summary; adds Brier and accuracy.",
}
SKLEARN_COSINE = {
    "kind": "code", "url": "https://github.com/scikit-learn/scikit-learn",
    "commit": "857849927da6e988d7d026b17aef214d43e5f26e",
    "path": "sklearn/metrics/pairwise.py", "symbol": "cosine_similarity",
    "license": "BSD-3-Clause", "adapted": "Pairwise cosine geometry for sample vectors.",
    "simplifications": "PyTorch CPU matrix multiplication replaces the NumPy/scikit-learn utility.",
}
TRL_GKD = {
    "kind": "code", "url": "https://github.com/huggingface/trl",
    "commit": "a7c34f363a8716473a0f15378621a3358b994417",
    "path": "trl/experimental/gkd/gkd_trainer.py", "symbol": "GKDTrainer.generalized_jsd_loss",
    "license": "Apache-2.0", "adapted": "Temperature-scaled full-vocabulary teacher-to-student KL.",
    "simplifications": "Fixes the divergence to forward KL and isolates masked logits from trainer/model plumbing.",
}
SLIME_OPD = {
    "kind": "code", "url": "https://github.com/THUDM/slime",
    "commit": "8ee9c1e1c8871ccd6dc8ec812edfaefa3dd1156b",
    "path": "slime/backends/megatron_utils/loss.py", "symbol": "apply_opd_kl_to_advantages",
    "license": "Apache-2.0", "adapted": "Subtracts sampled student-minus-teacher log-probability from advantages.",
    "simplifications": "Pure masked tensor interface rather than mutating a distributed RolloutBatch.",
}
OPSD = {
    "kind": "code", "url": "https://github.com/siyan-zhao/OPSD",
    "commit": "ae7d2519e94920c4eb6206c0c26de46d9c50abae",
    "path": "opsd_trainer.py", "symbol": "OPSDTrainer.generalized_jsd_loss",
    "license": "Apache-2.0 (source file header)", "adapted": "Full-vocabulary forward KL from a privileged-context teacher to the student.",
    "simplifications": "A tiny CPU model interface generates student tokens and scores both contexts; no distributed trainer, top-k, or pointwise clipping.",
}


def code_source(base, adapted, simplifications):
    return {**base, "adapted": adapted, "simplifications": simplifications}


def paper(url, section):
    return {"kind": "paper", "url": url, "section": section}


def case(name, code, behavior="rl.masking", *, unshown=False):
    result = {"name": name, "code": code, "behavior": behavior}
    if unshown:
        result.update(visibility="unshown", failure_message=f"{name}: check the stated contract and edge cases.")
    return result


def task(title, difficulty, function_name, description, prerequisites, hints, sources, tests, solution,
         *, model_connections, pros, cons, interview_questions=None):
    result = {
        "title": title, "difficulty": difficulty, "version": 1,
        "function_name": function_name, "description_en": description,
        "advisory_prerequisites": prerequisites,
        "hints": [
            {"level": 1, "kind": "questions", "content": hints[0]},
            {"level": 2, "kind": "analysis", "content": hints[1]},
        ],
        "model_connections": model_connections,
        "pro_con_analysis": {"pros": pros, "cons": cons},
        "sources": sources, "tests": tests, "solution": solution,
    }
    if interview_questions is not None:
        result["interview_questions"] = interview_questions
    return result
