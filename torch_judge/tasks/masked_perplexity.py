"""Compute response-only perplexity from token log probabilities."""

from ._interview import interview
from ._rl_extension import NANO, case, code_source, paper, task

TASK = task(
    "Masked Response Perplexity", "Medium", "masked_perplexity",
    """Implement `masked_perplexity(logprobs, mask) -> Tensor`. Both tensors have the same `[batch, time]` shape. `logprobs` contains log-probabilities of the realized next tokens; `mask` marks response tokens (1/True) and excludes prompt and padding (0/False). Return a scalar `exp(-sum(mask * logprobs) / sum(mask))`. This is token-weighted across the whole batch, not the mean of per-sequence perplexities. Reject shape mismatch or an all-zero mask with `ValueError`. Keep gradients to valid log-probabilities; masked positions have zero gradient. The caller has already aligned logits to next-token labels.""",
    ["per_token_logprobs", "response_token_mask"],
    ("Should a long response count more than a short one? What does perplexity become for constant log probability -ln(2)?",
     "Use the total number of valid tokens as denominator, then exponentiate the negative masked mean. Avoid averaging already exponentiated per-row values."),
    [code_source(NANO, "Realized-token log probabilities used to form a masked response NLL.", "Only the masked perplexity reduction remains."), paper("https://arxiv.org/abs/1904.09751", "Section 4.2: Perplexity")],
    [
        case("One valid token gives perplexity two", """
import torch, math
lp=torch.tensor([[-math.log(2),-99.0]])
assert torch.allclose({fn}(lp,torch.tensor([[1,0]])),torch.tensor(2.0))
"""),
        case("Seeded token-weighted oracle and gradient", """
import torch, math
torch.manual_seed(31)
for n,t in [(2,3),(4,6)]:
    lp=(-torch.rand(n,t)*3).requires_grad_()
    mask=torch.randint(0,2,(n,t)); mask[0,0]=1
    values=[-lp[i,j].item() for i in range(n) for j in range(t) if mask[i,j]]
    expected=math.exp(sum(values)/len(values))
    out={fn}(lp,mask)
    assert abs(out.item()-expected)<1e-5
    out.backward()
    assert torch.all(lp.grad[mask==0]==0)
    assert torch.all(lp.grad[mask==1]<0)
""", "gradient.flow", unshown=True),
        case("Rejects empty and shape mismatch", """
import torch
for lp,mask in [(torch.zeros(2,3),torch.zeros(2,3)),(torch.zeros(2,3),torch.ones(3,2))]:
    try: {fn}(lp,mask)
    except ValueError: pass
    else: raise AssertionError('expected ValueError')
""", "edge.empty_or_boundary", unshown=True),
        case("Nonfinite padding is excluded exactly", """
import torch, math
lp=torch.tensor([[-math.log(4),-float('inf')]],requires_grad=True)
out={fn}(lp,torch.tensor([[1,0]]))
assert torch.isfinite(out) and abs(out.item()-4)<1e-6
out.backward()
assert lp.grad[0,1]==0 and torch.isfinite(lp.grad).all()
""", "rl.masking", unshown=True),
    ],
    '''import torch

def masked_perplexity(logprobs, mask):
    if logprobs.shape != mask.shape:
        raise ValueError("shape mismatch")
    valid = mask.bool()
    if not valid.any():
        raise ValueError("no valid tokens")
    return torch.exp(-logprobs[valid].mean())
''',
    model_connections=["nano-aha-moment computes realized-token log probabilities; this task reduces those scores into response-only token-weighted perplexity."],
    pros=["A single scalar exposes deteriorating likelihood on the exact response tokens used for RL."],
    cons=["Token-weighted batch perplexity differs from the mean of per-sequence perplexities reported by some trainers."],
    interview_questions=interview(
        concept=[
            'What does perplexity measure, and why compute it only on response tokens?',
            'How does perplexity relate to cross-entropy?',
        ],
        deep_dive=[
            'Token-weighted across the batch versus mean of per-sequence perplexities: how do they differ numerically?',
            'Why exponentiate the mean negative log-prob instead of averaging per-token probabilities?',
            'What should happen with an all-zero mask?',
        ],
        tradeoffs=[
            'Why is perplexity a poor proxy for RL-trained model quality?',
            'How can two models with the same perplexity differ in downstream accuracy?',
        ],
    ),
)
