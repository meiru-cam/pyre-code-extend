"""Measure confidence quality for binary verifiable outcomes."""

from ._interview import interview
from ._rl_extension import TORCHMETRICS, case, paper, task

TASK = task(
    "RL Confidence and Calibration", "Medium", "confidence_calibration",
    """Implement `confidence_calibration(confidences, correct, bins=10) -> dict`. Inputs are equal-length one-dimensional tensors. `confidences[i]` is the probability assigned to the answer being correct, in [0,1]; `correct[i]` is 0/1 ground truth. Return Python floats `accuracy`, `brier`, and `ece`. Accuracy is mean correctness. Brier is mean squared `(confidence-correct)` error. ECE is sum over nonempty equal-width bins of `(bin_count/N) * abs(mean_confidence - mean_correct)`. Assign confidence 0 to the first bin and confidence 1 to the last; boundaries `k/bins` belong to the higher bin. Reject empty inputs, non-1D or mismatched shapes, invalid probabilities/labels, and nonpositive integer bins with `ValueError`. This binary calibration audit is not token-level entropy or a confidence score for a free-form response.""",
    ["rl_eval_loop"],
    ("Is an accurate model necessarily calibrated? Which bin contains confidence exactly 1.0?",
     "Compute Brier and accuracy across examples. For ECE, floor(confidence*bins) and cap at bins-1, then use bin-size weighting."),
    [TORCHMETRICS, paper("https://arxiv.org/abs/1706.04599", "Section 2: Expected Calibration Error and reliability diagrams")],
    [
        case("Hand-calculated two-bin example", """
import torch
out={fn}(torch.tensor([0.2,0.8]),torch.tensor([0,1]),bins=2)
assert abs(out['accuracy']-0.5)<1e-7
assert abs(out['brier']-0.04)<1e-6
assert abs(out['ece']-0.2)<1e-6
""", "numerics.stability"),
        case("Seeded independent bin oracle", """
import torch, random
random.seed(39)
for bins in (2,3,7):
    c=[random.random() for _ in range(31)]+[0.0,1.0]
    y=[random.randrange(2) for _ in c]
    groups=[[] for _ in range(bins)]
    for p,label in zip(c,y): groups[min(int(p*bins),bins-1)].append((p,label))
    ece=sum(len(g)/len(c)*abs(sum(p for p,_ in g)/len(g)-sum(z for _,z in g)/len(g)) for g in groups if g)
    out={fn}(torch.tensor(c),torch.tensor(y),bins)
    assert abs(out['accuracy']-sum(y)/len(y))<1e-6
    assert abs(out['brier']-sum((p-z)**2 for p,z in zip(c,y))/len(c))<1e-6
    assert abs(out['ece']-ece)<1e-6
""", "numerics.stability", unshown=True),
        case("Invalid inputs and bin boundary", """
import torch
for c,y,b in [(torch.tensor([]),torch.tensor([]),2),(torch.tensor([1.1]),torch.tensor([1]),2),(torch.tensor([0.5]),torch.tensor([2]),2),(torch.tensor([0.5]),torch.tensor([1]),0),(torch.ones(2),torch.ones(1),2)]:
    try: {fn}(c,y,b)
    except ValueError: pass
    else: raise AssertionError('expected ValueError')
out={fn}(torch.tensor([0.5,1.0]),torch.tensor([0,1]),2)
assert abs(out['ece']-0.25)<1e-6
""", "edge.empty_or_boundary", unshown=True),
    ],
    '''import torch

def confidence_calibration(confidences, correct, bins=10):
    if (confidences.ndim != 1 or correct.ndim != 1 or confidences.shape != correct.shape
            or confidences.numel() == 0 or not isinstance(bins, int) or isinstance(bins, bool) or bins <= 0):
        raise ValueError("invalid shape or bins")
    if (not torch.isfinite(confidences).all() or (confidences < 0).any() or (confidences > 1).any()
            or not torch.isfinite(correct).all() or ((correct != 0) & (correct != 1)).any()):
        raise ValueError("invalid probabilities or labels")
    labels = correct.to(confidences.dtype)
    bucket = (confidences * bins).long().clamp(max=bins - 1)
    ece = confidences.new_zeros(())
    for index in range(bins):
        chosen = bucket == index
        if chosen.any():
            ece = ece + chosen.float().mean() * (confidences[chosen].mean() - labels[chosen].mean()).abs()
    return {"accuracy": labels.mean().item(), "brier": ((confidences - labels) ** 2).mean().item(), "ece": ece.item()}
''',
    model_connections=["TorchMetrics' calibration-error implementation bins confidences and compares confidence with empirical correctness; this task adds binary Brier score and accuracy."],
    pros=["ECE and Brier expose overconfidence that accuracy alone hides."],
    cons=["ECE depends on bin count and these binary scores need a defensible correctness label."],
    interview_questions=interview(
        concept=[
            'What is calibration, and why does it matter for RL-trained reasoning models?',
            'How do accuracy, Brier score and ECE differ?',
        ],
        deep_dive=[
            'How do you assign confidences to equal-width bins, including the boundaries at 0 and 1?',
            "Why weight each bin's gap by its count?",
            'How do empty bins affect ECE?',
        ],
        tradeoffs=[
            'Why is ECE sensitive to the number of bins, and what alternatives exist?',
            'RL with correctness rewards often makes models overconfident. How would you fix that?',
        ],
    ),
)
