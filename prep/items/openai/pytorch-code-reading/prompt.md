The file below trains a classifier whose labels come from several annotators instead of a single ground truth. There are $N$ samples, each a feature vector in $\mathbb{R}^D$; the true class, in $\lbrace 0, \dots, C-1 \rbrace$, is never seen during training. There are $A$ annotators, and their labels are stored in an integer tensor `Y` of shape `(N, A)`: `Y[i, a]` is the class annotator `a` reported for sample `i`, or `-1` if annotator `a` never labeled sample `i` (every sample has at least 2 labels). Each annotator `a` has a *confusion matrix* $M_a \in \mathbb{R}^{C \times C}$: given that the true class is $c$, annotator `a` reports class $o$ with probability $M_a[c, o]$, so row $c$ of $M_a$ sums to 1. A reliable annotator's $M_a$ is close to the identity matrix; an unreliable one's is close to uniform in every row.

A small classifier, `Net`, maps `X` to class probabilities `p` of shape `(N, C)`. The confusion matrices are a second set of learnable parameters, held by `CrowdLayer`. Training never looks at the true class: for every observed pair `(i, a)`, the predicted distribution over what annotator `a` reports is `p[i] @ M[a]`, and the loss is the negative log-likelihood of the label `a` actually reported.

```python
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

N, D, C, A, H = 400, 10, 4, 5, 24
N_TEST = 200
K_RANGE = (2, 4)                            # each sample gets labeled by 2 or 3 annotators, chosen at random
RELIABILITY = [0.95, 0.8, 0.6, 0.4, 0.15]   # ground-truth diagonal of each annotator's confusion matrix

CENTERS = torch.randn(C, D, generator=torch.Generator().manual_seed(0)) * 2.0   # shared by every call

M_TRUE = torch.zeros(A, C, C)
for a in range(A):
    r = RELIABILITY[a]
    for c in range(C):
        row = torch.full((C,), (1 - r) / (C - 1))
        row[c] = r
        M_TRUE[a, c] = row


def make_data(seed, n=N):
    g = torch.Generator().manual_seed(seed)
    y = torch.randint(0, C, (n,), generator=g)
    X = CENTERS[y] + torch.randn(n, D, generator=g) * 0.8

    Y = torch.full((n, A), -1, dtype=torch.long)
    for i in range(n):
        k = torch.randint(K_RANGE[0], K_RANGE[1], (1,), generator=g).item()
        for a in torch.randperm(A, generator=g)[:k].tolist():
            Y[i, a] = torch.multinomial(M_TRUE[a, y[i]], 1, generator=g).item()
    return X, y, Y


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(D, H)
        self.l2 = nn.Linear(H, C)

    def forward(self, x):
        return self.l2(F.relu(self.l1(x)))


class CrowdLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.logits = nn.Parameter(torch.eye(C).unsqueeze(0).repeat(A, 1, 1) * 2.0)

    def forward(self):
        return F.softmax(self.logits, dim=-1)


def nll_loop(p, M, Y):
    # p: class probabilities from the classifier. M: annotator confusion matrices. Y: observed labels.
    total = torch.zeros(())
    for i in range(Y.shape[0]):
        for a in range(Y.shape[1]):
            lab = Y[i, a].item()
            if lab < 0:
                continue
            q = p[i] @ M[a]
            total = total + -torch.log(q[lab] + 1e-8)
    return total


def raw_annotator_accuracy(Y, y):
    # how often each annotator's own label matches the true class, wherever they voted; only
    # computable offline for evaluation, since y is never seen during training
    acc = torch.zeros(A)
    for a in range(A):
        voted = Y[:, a] >= 0
        acc[a] = (Y[voted, a] == y[voted]).float().mean()
    return acc


def run(seed=0, epochs=25, lr=5e-2):
    torch.manual_seed(seed)                 # reseeds Net's and CrowdLayer's weight init too
    X, y, Y = make_data(seed)
    X_test, y_test, _ = make_data(seed + 1000, n=N_TEST)
    net, crowd = Net(), CrowdLayer()
    opt = torch.optim.Adam(list(net.parameters()) + list(crowd.parameters()), lr=lr)
    n_votes = (Y >= 0).sum().item()

    t_start = time.time()
    for ep in range(epochs):
        opt.zero_grad()
        p = F.softmax(net(X), dim=-1)
        M = crowd()
        loss = nll_loop(p, M, Y) / n_votes
        loss.backward()
        opt.step()
        with torch.no_grad():
            train_probs = F.softmax(net(X), dim=-1)          # recomputed for logging, not reused from p
            train_acc = (train_probs.argmax(-1) == y).float().mean().item()
        if ep % 5 == 0 or ep == epochs - 1:
            print(f"epoch {ep:2d}  loss {loss.item():.4f}  train_acc {train_acc:.3f}")
    train_seconds = time.time() - t_start

    with torch.no_grad():
        test_acc = (net(X_test).argmax(-1) == y_test).float().mean().item()
    print(f"test_acc {test_acc:.3f}  ({epochs} epochs, {train_seconds:.1f}s)")
    raw_acc = [round(v, 2) for v in raw_annotator_accuracy(Y, y).tolist()]
    print("raw annotator accuracy vs ground truth:", raw_acc)
    print("ground-truth annotator reliability:     ", RELIABILITY)
    return net, crowd, X, Y


torch.manual_seed(0)
net, crowd, X, Y = run()
```

Part 1 asks about the code above. Parts 2 and 3 extend and refactor it.

### Part 1 — Reading the code: data flow, shapes, and complexity

Answer the following, using $N, D, C, A, H$ for the constants the file defines and $P$ for `(Y >= 0).sum()`, the number of observed (sample, annotator) pairs.

- Trace the data from `X` to the update applied by `opt.step()`: which functions and tensors does it pass through, and at what point, if any, does `y` enter?
- What are the shapes of `net(X)`, `p = F.softmax(net(X), dim=-1)`, and `crowd()`?
- Inside `nll_loop`, `q = p[i] @ M[a]` is one matrix product. What are the shapes of its two operands? What is its time complexity, and how much memory does `q` occupy, in terms of $C$?
- The matrix products inside `net.l1` and `net.l2` — what is the time complexity and forward-output memory of each, in terms of $N, D, H, C$? What extra tensors does the backward pass need to keep around?
- What is the overall time complexity of `nll_loop`? Give a loose upper bound in terms of $N, A, C$, and a tighter one in terms of $P$.
- `nll_loop` itself never materializes a tensor of size $O(N \cdot A \cdot C)$. If instead you computed `p[i] @ M[a]` for every $(i, a)$ pair at once with `einsum`, before masking out the missing ones, how much memory would that intermediate tensor use? Can this be avoided?
- In one training step, how many multiply-adds does the forward pass of `net` do, compared to `nll_loop`? Which of the two actually takes longer to run, and why don't the two answers agree?

### Part 2 — Extension: annotator reliability from the learned confusion matrices

Add a method to `CrowdLayer` that estimates each annotator's reliability directly from its learned confusion matrix, and a function that picks out the least reliable one. An annotator's reliability is the average, over the $C$ classes, of the probability that it reports the true class: $\frac{1}{C}\sum_c M_a[c, c]$. Implement `reliability` as a free function and attach it with `CrowdLayer.reliability = reliability`.

```py
def reliability(self) -> torch.Tensor:
    """Attached as CrowdLayer.reliability. Returns r of shape (A,), r[a] in [0, 1]: the estimated
    reliability of annotator a, computed from self.forward(), the learned confusion matrices."""


def least_reliable(r: torch.Tensor) -> int:
    """r: (A,). Returns the index of the annotator with the lowest estimated reliability."""
```

The data above was generated with a fixed ground-truth reliability for every annotator, in `RELIABILITY`; use it to check that `least_reliable` finds the right one.

### Part 3 — Refactor: vectorizing `nll_loop`

`nll_loop` computes the loss with a Python loop over every sample and, inside it, another loop over every annotator. Rewrite it as `nll_vectorized`, with no Python loop over samples or annotators — indexing, `gather`, `einsum`, or batched matrix multiplication (`torch.bmm`) — returning the same scalar (up to floating-point error) as `nll_loop`, for any `p`, `M`, `Y` of matching shapes.

```py
def nll_vectorized(p: torch.Tensor, M: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """p: (N, C) class probabilities. M: (A, C, C) confusion matrices. Y: (N, A) observed labels,
    -1 for missing. Returns the same scalar as nll_loop(p, M, Y)."""
```

### Bonus — Complexity of three more operators

Answer the following without reference to the code above; give the conclusion and a one-line reason for each.

- Self-attention over a sequence of length $L$ with hidden dimension $d$, split into $h$ heads of dimension $d / h$ each. What is the time complexity of one forward pass? What term dominates the memory needed to run backward through it?
- A 2D convolution layer: input $(B, C_{in}, H_{in}, W_{in})$, kernel $k \times k$, $C_{out}$ output channels, output spatial size $H_{out} \times W_{out}$. What is the time complexity of one forward pass?
- Batched matrix multiplication `(B, n, m) @ (B, m, p)`: what is its time complexity, and how much memory does the output occupy?
