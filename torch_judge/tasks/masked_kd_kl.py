"""Temperature-scaled masked knowledge-distillation loss."""

from ._interview import interview
from ._rl_extension import TRL_GKD, case, paper, task

TASK = task(
    "Masked Temperature KD", "Medium", "masked_kd_kl",
    """Implement `masked_kd_kl(student_logits, teacher_logits, mask, temperature=1.0) -> Tensor`. Logits have matching `[batch,time,vocab]` shape; mask is `[batch,time]`. At each valid token compute full-vocabulary forward KL `KL(softmax(teacher/T) || softmax(student/T))`, average across all valid tokens, then multiply by `T*T`. Reject invalid shapes, no valid tokens, or nonpositive/nonfinite temperature with `ValueError`. Detach teacher logits so gradients update only the student. Masked positions must receive zero gradient. This is teacher-forced distribution matching; it does not itself make rollouts on-policy.""",
    ["per_token_logprobs", "response_token_mask"],
    ("Which distribution weights the log-ratio in forward KL? Why multiply by temperature squared?",
     "Apply log_softmax to both logits divided by T, take teacher probability times teacher-minus-student log probability, mask and token-average, then scale by T squared. Detach the teacher first."),
    [TRL_GKD, paper("https://arxiv.org/abs/1503.02531", "Section 2: Distillation and temperature scaling")],
    [
        case("Visible nonzero temperature-scaled forward KL", """
import torch, math
student=torch.tensor([[[0.,0.]]])
teacher=torch.tensor([[[2*math.log(3),0.]]])
expected=4*(.75*math.log(.75/.5)+.25*math.log(.25/.5))
assert abs({fn}(student,teacher,torch.tensor([[1]]),2.0).item()-expected)<1e-6
""", "rl.kl_estimator"),
        case("Equal distributions have zero KL", """
import torch
x=torch.tensor([[[1.,2.],[-1.,3.]]])
out={fn}(x,x.clone(),torch.tensor([[1,0]]),2.0)
assert abs(out.item())<1e-6
""", "rl.kl_estimator"),
        case("Seeded independent softmax oracle and gradient isolation", """
import torch, math
torch.manual_seed(51)
for temp in (0.5,1.0,2.0):
    s=torch.randn(2,3,4,requires_grad=True)
    t=torch.randn(2,3,4,requires_grad=True)
    mask=torch.tensor([[1,0,1],[0,1,0]])
    terms=[]
    for i,j in [(0,0),(0,2),(1,1)]:
        a=(t[i,j].detach()/temp).tolist(); b=(s[i,j].detach()/temp).tolist()
        pa=[math.exp(z-max(a)) for z in a]; pa=[z/sum(pa) for z in pa]
        pb=[math.exp(z-max(b)) for z in b]; pb=[z/sum(pb) for z in pb]
        terms.append(sum(p*math.log(p/q) for p,q in zip(pa,pb)))
    out={fn}(s,t,mask,temp)
    assert abs(out.item()-sum(terms)/3*temp*temp)<1e-5
    out.backward()
    assert t.grad is None or torch.all(t.grad==0)
    assert torch.all(s.grad[mask==0]==0)
    assert torch.any(s.grad[mask==1]!=0)
""", "gradient.flow", unshown=True),
        case("Invalid shape, mask, and temperature", """
import torch
s=torch.zeros(1,2,3)
for t,m,temp in [(s,torch.zeros(1,2),1),(s,torch.ones(1,2),0),(torch.zeros(1,2,4),torch.ones(1,2),1),(s,torch.ones(2,1),1)]:
    try: {fn}(s,t,m,temp)
    except ValueError: pass
    else: raise AssertionError('expected ValueError')
""", "edge.empty_or_boundary", unshown=True),
        case("Masked invalid logits have no effect", """
import torch
s=torch.tensor([[[0.,1.],[float('-inf'),float('-inf')]]],requires_grad=True)
t=torch.tensor([[[1.,0.],[float('-inf'),float('-inf')]]])
out={fn}(s,t,torch.tensor([[1,0]]))
assert torch.isfinite(out)
out.backward()
assert torch.isfinite(s.grad).all() and torch.all(s.grad[0,1]==0)
""", "rl.masking", unshown=True),
    ],
    '''import math
import torch

def masked_kd_kl(student_logits, teacher_logits, mask, temperature=1.0):
    if (student_logits.ndim != 3 or student_logits.shape != teacher_logits.shape
            or mask.shape != student_logits.shape[:2] or student_logits.shape[-1] < 2
            or not math.isfinite(float(temperature)) or temperature <= 0 or not mask.bool().any()):
        raise ValueError("invalid logits, mask, or temperature")
    valid = mask.bool()
    student_logp = torch.log_softmax(student_logits[valid] / temperature, dim=-1)
    teacher_logp = torch.log_softmax(teacher_logits.detach()[valid] / temperature, dim=-1)
    teacher_prob = teacher_logp.exp()
    per_token = (teacher_prob * (teacher_logp - student_logp)).sum(dim=-1)
    return per_token.mean() * temperature ** 2
''',
    model_connections=["TRL GKD computes temperature-scaled token distribution divergence between a teacher and student; this exercise fixes the divergence to forward KL."],
    pros=["Full-vocabulary targets transmit more information than hard teacher labels."],
    cons=["Teacher-forced tokens need not match the student's rollout distribution and full logits cost memory."],
    interview_questions=interview(
        concept=[
            "When training a small model or compressing an LLM, which distillation methods have you used?",
            "For sequence- or token-level knowledge distillation, how do forward KL and reverse KL differ, and what behavior does each encourage in the student?",
            "Why divide the logits by a temperature T when computing the KD loss, and why multiply the loss by T*T at the end?",
        ],
        deep_dive=[
            "What goes wrong if padding or masked positions contribute to the loss or the gradient? How do you handle the mask and numerical stability in code?",
            "Should gradients flow back into the teacher? How do you cut them in PyTorch?",
            "If every position in the batch is masked, or T <= 0, what should the loss function do and why?",
        ],
        tradeoffs=[
            "Temperature KD is teacher-forced distillation. What are its limits for autoregressive generation, for example exposure bias or distribution shift?",
            "Compared with plain cross-entropy SFT on teacher outputs, what does matching the full teacher distribution buy you, and what does it cost?",
        ],
    ),
)
