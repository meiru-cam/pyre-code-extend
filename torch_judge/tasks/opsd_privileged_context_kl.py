"""Full-vocabulary OPSD teacher-to-student KL on student trajectories."""

from ._rl_extension import OPSD, case, paper, task

TASK = task(
    "OPSD Privileged-Context KL", "Hard", "opsd_privileged_context_kl",
    """Implement `opsd_privileged_context_kl(model, problems, gold_solutions, max_new_tokens) -> Tensor` on a tiny offline model interface.

`problems` and `gold_solutions` are equal-length nonempty lists. Call `model.generate(problems, max_new_tokens)` **without the gold solutions**; it returns `(tokens, mask)` with matching `[batch,time]` shapes. Then call the *same model object* twice on those exact generated tokens: `model.logits(problems, tokens, privileged_context=None)` for the student and `model.logits(problems, tokens, privileged_context=gold_solutions)` for the privileged teacher. Each call returns aligned `[batch,time,vocab]` logits. The teacher must not generate another trajectory. At every valid token compute the full-vocabulary forward KL `KL(teacher || student)` and return its valid-token mean. Detach teacher logits; only the student score branch receives gradients. Reject empty or mismatched input lists, nonpositive `max_new_tokens`, malformed token/mask/logit shapes, vocab smaller than two, or an all-zero mask with `ValueError`. Padding scores are excluded before arithmetic. Unlike sampled OPD, every vocabulary probability contributes; unlike temperature KD, no temperature scaling is used.""",
    ["masked_kd_kl", "opd_sampled_kl_advantage"],
    ("Which context may the generation call see? Can the teacher choose a different trajectory? Which score branch receives gradients?",
     "Generate once from problem-only context, then reuse those tokens in two score calls on the same model. Pass gold solutions only to the teacher call, detach that branch, and average teacher-weighted full-vocabulary KL over valid positions."),
    [OPSD, paper("https://arxiv.org/abs/2601.18734", "Section 3: On-policy self-distillation; Section 4.3.5: full-vocabulary objective")],
    [
        case("Visible nonzero KL on student-generated tokens", """
import torch, math
class TinyModel:
    def generate(self, problems, max_new_tokens):
        assert problems == ['2+2'] and max_new_tokens == 2
        return torch.tensor([[4, 0]]), torch.tensor([[1, 0]])
    def logits(self, problems, tokens, privileged_context=None):
        assert problems == ['2+2'] and tokens.tolist() == [[4, 0]]
        return (torch.tensor([[[math.log(3), 0.], [0., 0.]]]) if privileged_context == ['4']
                else torch.tensor([[[0., 0.], [0., 0.]]]))
out = {fn}(TinyModel(), ['2+2'], ['4'], 2)
expected = .75 * math.log(.75 / .5) + .25 * math.log(.25 / .5)
assert abs(out.item() - expected) < 1e-6
""", "rl.kl_estimator"),
        case("Generation and both score calls use the required contexts", """
import torch, math
class RecordingModel:
    def __init__(self):
        self.calls=[]
        self.student=torch.tensor([[[0.,0.]]],requires_grad=True)
        self.teacher=torch.tensor([[[math.log(3),0.]]],requires_grad=True)
    def generate(self, problems, max_new_tokens):
        self.calls.append(('generate',list(problems),max_new_tokens))
        return torch.tensor([[7]]),torch.tensor([[1]])
    def logits(self, problems, tokens, privileged_context=None):
        self.calls.append(('logits',list(problems),tokens.tolist(),privileged_context))
        return self.student if privileged_context is None else self.teacher
m=RecordingModel()
out={fn}(m,['question'],['answer'],1)
assert m.calls==[('generate',['question'],1),('logits',['question'],[[7]],None),('logits',['question'],[[7]],['answer'])]
out.backward()
assert m.student.grad is not None and torch.any(m.student.grad!=0)
assert m.teacher.grad is None or torch.all(m.teacher.grad==0)
""", "gradient.flow", unshown=True),
        case("Seeded independent full-vocabulary oracle", """
import torch, math
torch.manual_seed(71)
s=torch.randn(2,3,5,dtype=torch.float64)
t=torch.randn(2,3,5,dtype=torch.float64)
mask=torch.tensor([[1,1,0],[0,1,0]])
tokens=torch.tensor([[2,3,0],[1,0,0]])
class FixedModel:
    def generate(self, problems, max_new_tokens): return tokens,mask
    def logits(self, problems, actual_tokens, privileged_context=None):
        assert torch.equal(actual_tokens,tokens)
        return s if privileged_context is None else t
total=0.
for i,j in [(0,0),(0,1),(1,1)]:
    a=t[i,j].tolist(); b=s[i,j].tolist()
    p=[math.exp(z-max(a)) for z in a]; p=[z/sum(p) for z in p]
    q=[math.exp(z-max(b)) for z in b]; q=[z/sum(q) for z in q]
    total+=sum(x*math.log(x/y) for x,y in zip(p,q))
assert abs({fn}(FixedModel(),['a','b'],['A','B'],3).item()-total/3)<1e-12
""", "numerics.stability", unshown=True),
        case("Rejects invalid generated and scored shapes", """
import torch
class BadModel:
    def __init__(self,mask,teacher_shape=(1,2,3)): self.mask=mask; self.teacher_shape=teacher_shape
    def generate(self,problems,max_new_tokens): return torch.ones(1,2,dtype=torch.long),self.mask
    def logits(self,problems,tokens,privileged_context=None):
        return torch.zeros(self.teacher_shape if privileged_context is not None else (1,2,3))
for args in [(BadModel(torch.zeros(1,2)),['q'],['a'],2),(BadModel(torch.ones(2,1)),['q'],['a'],2),(BadModel(torch.ones(1,2),(1,2,4)),['q'],['a'],2),(BadModel(torch.ones(1,2)),[],[],2),(BadModel(torch.ones(1,2)),['q'],['a'],0)]:
    try: {fn}(*args)
    except ValueError: pass
    else: raise AssertionError('expected ValueError')
""", "edge.empty_or_boundary", unshown=True),
        case("Masked invalid logits have no effect", """
import torch
class Model:
    def __init__(self): self.s=torch.tensor([[[0.,1.],[float('-inf'),float('-inf')]]],requires_grad=True)
    def generate(self,problems,max_new_tokens): return torch.tensor([[1,0]]),torch.tensor([[1,0]])
    def logits(self,problems,tokens,privileged_context=None):
        return self.s if privileged_context is None else torch.tensor([[[1.,0.],[float('-inf'),float('-inf')]]])
m=Model()
out={fn}(m,['q'],['a'],2)
assert torch.isfinite(out)
out.backward()
assert torch.isfinite(m.s.grad).all() and torch.all(m.s.grad[0,1]==0)
""", "rl.masking", unshown=True),
    ],
    '''import torch

def opsd_privileged_context_kl(model, problems, gold_solutions, max_new_tokens):
    if (len(problems) == 0 or len(problems) != len(gold_solutions)
            or not isinstance(max_new_tokens, int) or isinstance(max_new_tokens, bool)
            or max_new_tokens <= 0):
        raise ValueError("invalid problems or generation limit")
    tokens, mask = model.generate(problems, max_new_tokens)
    if (tokens.ndim != 2 or mask.shape != tokens.shape or tokens.shape[0] != len(problems)
            or not mask.bool().any()):
        raise ValueError("invalid generated batch or mask")
    student_logits = model.logits(problems, tokens, privileged_context=None)
    privileged_teacher_logits = model.logits(problems, tokens, privileged_context=gold_solutions)
    if (student_logits.ndim != 3 or student_logits.shape != privileged_teacher_logits.shape
            or student_logits.shape[:2] != tokens.shape or student_logits.shape[-1] < 2):
        raise ValueError("invalid score shapes")
    valid = mask.bool()
    student_logp = torch.log_softmax(student_logits[valid], dim=-1)
    teacher_logp = torch.log_softmax(privileged_teacher_logits.detach()[valid], dim=-1)
    per_token = (teacher_logp.exp() * (teacher_logp - student_logp)).sum(dim=-1)
    return per_token.mean()
''',
    model_connections=["The OPSD author implementation scores student-generated tokens twice with one model: ordinary student context and a privileged gold-solution teacher context."],
    pros=["The privileged branch supplies dense token-level guidance without a separate teacher checkpoint."],
    cons=["Two full-vocabulary score passes cost memory and gold-solution context must never leak into generation."],
)
