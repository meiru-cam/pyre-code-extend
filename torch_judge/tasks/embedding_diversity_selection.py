"""Greedy max-min selection in embedding space."""

from ._interview import interview
from ._rl_extension import SKLEARN_COSINE, case, code_source, paper, task

TASK = task(
    "Embedding-Space Data Diversity", "Hard", "embedding_diversity_selection",
    """Implement `embedding_diversity_selection(embeddings, k, seed_index=0) -> list[int]`. `embeddings` is a finite, nonzero-row float tensor `[N,D]`. Select `k` distinct row indices with greedy farthest-first traversal under cosine distance `1 - cosine_similarity`. Start with `seed_index`. At each step choose the remaining row whose distance to its *nearest selected row* is greatest (max-min diversity). Break exact ties by lower row index. Return indices in selection order. Reject empty/invalid shape, zero vectors, nonfinite entries, invalid `k`, and out-of-range seed with `ValueError`. This measures representation diversity, not gradient orthogonality or train/eval leakage.""",
    ["rl_data_audit"],
    ("Why use the minimum distance to already-selected examples instead of mean distance? How should exact ties resolve?",
     "Normalize each row, form cosine distances, and greedily maximize each candidate's minimum distance to the selected set. Keep selection order explicit."),
    [code_source(SKLEARN_COSINE, "Cosine matrix used by farthest-first selection; the greedy rule follows the cited core-set paper.", "PyTorch tensor implementation with a fixed seed and tie rule."), paper("https://arxiv.org/abs/1708.00489", "Section 2: Core-set selection with a k-center objective")],
    [
        case("Opposite vector is selected first", """
import torch
x=torch.tensor([[1.,0.],[-1.,0.],[0.,1.]])
assert {fn}(x,3)==[0,1,2]
""", "routing.selection"),
        case("Seeded independent greedy cosine oracle", """
import torch, math
torch.manual_seed(83)
for n,d,k in [(5,3,3),(9,4,5)]:
    x=torch.randn(n,d)
    rows=x.tolist()
    def dist(i,j):
        a,b=rows[i],rows[j]
        return 1-sum(u*v for u,v in zip(a,b))/(math.sqrt(sum(u*u for u in a))*math.sqrt(sum(v*v for v in b)))
    expected=[2]
    while len(expected)<k:
        expected.append(max((i for i in range(n) if i not in expected),key=lambda i:(min(dist(i,j) for j in expected),-i)))
    assert {fn}(x,k,2)==expected
""", "routing.selection", unshown=True),
        case("Ties and invalid rows", """
import torch
x=torch.tensor([[1.,0.],[0.,1.],[0.,-1.]])
assert {fn}(x,2)==[0,1]
for y,k,s in [(torch.zeros(2,2),1,0),(torch.ones(2,2),3,0),(torch.ones(2,2),1,2),(torch.ones(0,2),1,0)]:
    try: {fn}(y,k,s)
    except ValueError: pass
    else: raise AssertionError('expected ValueError')
""", "edge.empty_or_boundary", unshown=True),
    ],
    '''import torch
import torch.nn.functional as F

def embedding_diversity_selection(embeddings, k, seed_index=0):
    if (embeddings.ndim != 2 or embeddings.shape[0] == 0 or embeddings.shape[1] == 0
            or not torch.is_floating_point(embeddings) or not torch.isfinite(embeddings).all()
            or (embeddings.norm(dim=1) == 0).any() or not isinstance(k, int) or isinstance(k, bool)
            or k < 1 or k > embeddings.shape[0] or not isinstance(seed_index, int)
            or isinstance(seed_index, bool) or seed_index < 0 or seed_index >= embeddings.shape[0]):
        raise ValueError("invalid embeddings or selection size")
    normalized = F.normalize(embeddings, dim=1)
    distances = 1 - normalized @ normalized.T
    selected = [seed_index]
    while len(selected) < k:
        candidate = max((i for i in range(len(embeddings)) if i not in selected),
                        key=lambda i: (min(distances[i, j].item() for j in selected), -i))
        selected.append(candidate)
    return selected
''',
    model_connections=["scikit-learn's cosine_similarity supplies the geometry; the core-set k-center objective supplies the farthest-first selection rule."],
    pros=["Greedy max-min selection avoids choosing many near-duplicate embeddings."],
    cons=["The chosen seed and embedding quality can dominate the selected subset; cosine diversity is not gradient diversity."],
    interview_questions=interview(
        concept=[
            'Why select diverse training data, and what does embedding diversity capture?',
            'What is farthest-first traversal?',
        ],
        deep_dive=[
            "How do you maintain each row's distance to its nearest selected row efficiently?",
            'Why cosine distance rather than Euclidean for embeddings?',
            'How do tie-breaking and the seed index affect determinism?',
        ],
        tradeoffs=[
            'Diverse in embedding space does not mean useful for training. What else would you consider?',
            'Farthest-first tends to pick outliers. How do you protect against selecting noise?',
        ],
    ),
)
