"""Measure whether sample gradients reinforce or conflict."""

from ._rl_extension import SKLEARN_COSINE, case, code_source, paper, task

TASK = task(
    "Gradient Orthogonality Audit", "Hard", "gradient_orthogonality_audit",
    """Implement `gradient_orthogonality_audit(gradients) -> dict` for a finite float tensor `[N,D]` of flattened per-example gradients, with `N >= 2` and every row nonzero. Return `cosine`: the symmetric `[N,N]` pairwise cosine-similarity tensor (diagonal one); `conflict_pairs`: number of unordered off-diagonal pairs with cosine strictly below zero; and `max_abs_overlap`: largest absolute off-diagonal cosine as a Python float. Near-zero cosine means orthogonal update directions, while a negative value indicates opposing directions. Do not confuse this with embedding-space diversity. Reject invalid shape, nonfinite values, or zero gradients with `ValueError`. Preserve input dtype; no gradient backpropagation through this diagnostic is required. This is a per-example pairwise audit of the gradient-interference geometry discussed by PCGrad, not PCGrad's task-level projection algorithm.""",
    ["rl_data_audit"],
    ("What does cosine -1 mean for two training examples? Should diagonal entries count as overlaps?",
     "Normalize rows, multiply by their transpose, then inspect only upper-triangular off-diagonal entries for conflict and overlap."),
    [code_source(SKLEARN_COSINE, "Pairwise cosine matrix applied to flattened per-example gradients.", "PyTorch tensors rather than scikit-learn; PCGrad's task-level projection is not implemented."), paper("https://arxiv.org/abs/2001.06782", "Section 2: conflicting gradients and cosine similarity")],
    [
        case("Orthogonal and conflicting gradients", """
import torch
x=torch.tensor([[1.,0.],[0.,1.],[-1.,0.]])
out={fn}(x)
assert out['conflict_pairs']==1
assert abs(out['max_abs_overlap']-1)<1e-6
assert torch.allclose(out['cosine'],torch.tensor([[1.,0.,-1.],[0.,1.,0.],[-1.,0.,1.]]),atol=1e-6)
""", "numerics.stability"),
        case("Independent seeded pairwise oracle", """
import torch, math
torch.manual_seed(61)
for n,d in [(3,4),(7,6)]:
    x=torch.randn(n,d,dtype=torch.float64)
    out={fn}(x)
    rows=x.tolist()
    expected=torch.tensor([[sum(a*b for a,b in zip(u,v))/(math.sqrt(sum(a*a for a in u))*math.sqrt(sum(b*b for b in v))) for v in rows] for u in rows],dtype=x.dtype)
    assert out['cosine'].dtype==x.dtype
    assert torch.allclose(out['cosine'],expected,atol=1e-12)
    pairs=[expected[i,j].item() for i in range(n) for j in range(i+1,n)]
    assert out['conflict_pairs']==sum(z<0 for z in pairs)
    assert abs(out['max_abs_overlap']-max(abs(z) for z in pairs))<1e-12
""", "numerics.stability", unshown=True),
        case("Rejects zero and nonfinite gradients", """
import torch
for x in [torch.zeros(2,3),torch.ones(1,3),torch.ones(2,0),torch.tensor([[1.,0.],[float('nan'),1.]])]:
    try: {fn}(x)
    except ValueError: pass
    else: raise AssertionError('expected ValueError')
""", "edge.empty_or_boundary", unshown=True),
    ],
    '''import torch
import torch.nn.functional as F

def gradient_orthogonality_audit(gradients):
    if (gradients.ndim != 2 or gradients.shape[0] < 2 or gradients.shape[1] == 0
            or not torch.is_floating_point(gradients) or not torch.isfinite(gradients).all()
            or (gradients.norm(dim=1) == 0).any()):
        raise ValueError("invalid gradients")
    normalized = F.normalize(gradients, dim=1)
    cosine = normalized @ normalized.T
    cosine.diagonal().fill_(1)
    upper = cosine[torch.triu_indices(len(gradients), len(gradients), offset=1).unbind()]
    return {"cosine": cosine, "conflict_pairs": int((upper < 0).sum().item()),
            "max_abs_overlap": upper.abs().max().item()}
''',
    model_connections=["scikit-learn supplies pairwise cosine geometry; PCGrad motivates treating negative gradient similarity as interference, though this task only audits per-example gradients."],
    pros=["Separates aligned, orthogonal, and opposing sample updates before deciding how to mix training data."],
    cons=["Per-example gradients are costly to obtain, and pairwise cosine alone does not predict final training benefit."],
)
