Worth confirming with the interviewer: whether an annotator's reliability should be a single scalar averaged over classes, as Part 2 defines it, or a full per-class breakdown. Read `run` once, top to bottom, noting which tensors the optimizer updates and which are constants, before answering Part 1.

### Part 1

- `make_data` builds `X` (`(N, D)`), the observed labels `Y` (`(N, A)`), and uses the fixed `M_TRUE`; `y`, the true class, only reappears afterwards in `train_acc` and `test_acc`, and never touches the loss. Inside `run`, `X` goes through `net` to logits, then `softmax` to `p` (`(N, C)`); `crowd()` produces `M` (`(A, C, C)`); `nll_loop(p, M, Y)` combines `p`, `M`, and `Y` into the scalar `loss`; `loss.backward()` writes gradients into the parameters of both `net` and `crowd`, and `opt.step()` applies them.
- `net(X)`: `(N, C) = (400, 4)`. `p`: the same shape, `(N, C) = (400, 4)`. `crowd()`: `(A, C, C) = (5, 4, 4)`.
- `p[i]` has shape `(C,)`, `M[a]` has shape `(C, C)`: a `(m,) @ (m, k)` vector-matrix product with $m = k = C$, costing $O(C^2)$ time; the result `q` is a length-$C$ vector, $O(C)$ memory.
- `net.l1`: `(N, D) @ (D, H)`, time $O(NDH)$, forward output `(N, H)`, memory $O(NH)$. `net.l2`: `(N, H) @ (H, C)`, time $O(NHC)$, output `(N, C)`, memory $O(NC)$. Backward keeps `l1`'s input (`X`, $O(ND)$), the `ReLU` output ($O(NH)$; it is `l2`'s input and also gates the `ReLU` gradient, so the pre-activation itself is not kept) and the softmax output `p` ($O(NC)$): the same order as the forward activations.
- Loose bound $O(NAC^2)$: the double loop runs up to $N \cdot A$ times, and every iteration that does not `continue` does an $O(C^2)$ vector-matrix product. Since it `continue`s whenever `lab < 0`, the tight bound is $O(PC^2)$, with $P$ the number of iterations that actually reach the product. Here $N \cdot A = 2000$ and $P = 998$ — about a factor of 2 apart.
- Computing `p[i] @ M[a]` for every pair at once, e.g. with `einsum('nc,acd->nad', p, M)`, produces a tensor of shape `(N, A, C)`, $O(NAC)$ memory. At this scale that is $2000 \times 4 = 8000$ floats, nothing to worry about, but the ratio to the $O(PC)$ actually needed grows with $N \cdot A / P$: with many more annotators than any one item receives, most of that grid holds pairs with no label at all. It is avoidable — select the $P$ valid pairs before doing any arithmetic, which is what Part 3 does.
- `net`'s forward does about $NDH + NHC = 134{,}400$ multiply-adds; `nll_loop`'s tight bound is $PC^2 = 998 \times 16 = 15{,}968$, about $8\times$ fewer. Measured on one training step, though, `net`'s forward takes about 0.4 ms, while `nll_loop` takes about 34 ms to produce the loss and about 67 ms to run backward through it — `nll_loop` dominates by two orders of magnitude despite doing less arithmetic. The reason: its $P$ products are $P$ separate Python-level tensor calls, each paying a fixed interpreter and autograd bookkeeping cost, while `net`'s two linear layers are two calls into a single batched matrix-multiply kernel. Multiply-add counts and wall-clock time are not the same thing.

### Part 2

An annotator's confusion matrix is a stochastic matrix whose row $c$ is its reporting distribution given that the true class is $c$; the diagonal entry $M_a[c, c]$ is the probability it gets class $c$ right, so averaging the diagonal gives one reliability number per annotator, without ever looking at `y`.

```python
def reliability(self):
    return self.forward().diagonal(dim1=-2, dim2=-1).mean(-1)


CrowdLayer.reliability = reliability          # attach to the CrowdLayer class given in the problem


def least_reliable(r):
    return int(r.argmin().item())
```

On the trained `crowd` of the problem statement, `crowd.reliability()` is about `[0.94, 0.82, 0.58, 0.31, 0.26]`, against the ground truth `RELIABILITY = [0.95, 0.8, 0.6, 0.4, 0.15]`: the ranking matches, `least_reliable` returns `4`, and the same holds for three more seeds trained for fewer epochs. The last two estimates (0.31 and 0.26) lie closer together than the true values (0.4 and 0.15) — an annotator who rarely labels and reports close to uniformly gives the corresponding row of $M_a$ little evidence to learn from — but the ordering `least_reliable` needs is preserved.

### Part 3

Idea: instead of a Python loop over $(i, a)$, take the $P$ valid pairs at once with `nonzero`, gather the corresponding rows of `p` and matrices of `M`, and run all $P$ vector-matrix products as one batched matrix multiply.

```python
def nll_vectorized(p, M, Y):
    idx_i, idx_a = (Y >= 0).nonzero(as_tuple=True)                    # P pairs, P = (Y >= 0).sum()
    lab = Y[idx_i, idx_a]
    q = torch.bmm(p[idx_i].unsqueeze(1), M[idx_a]).squeeze(1)         # (P, 1, C) @ (P, C, C) -> (P, C)
    return -torch.log(q.gather(1, lab.unsqueeze(1)).squeeze(1) + 1e-8).sum()
```

`p[idx_i]` and `M[idx_a]` are advanced indexing, producing new tensors of shape `(P, C)` and `(P, C, C)`; `torch.bmm` treats the leading dimension as a batch of $P$ independent `(1, C) @ (C, C)` products; `gather` reads off, from each of the $P$ rows of `q`, the probability of the label that annotator actually reported. The asymptotic time is the same $O(PC^2)$ as `nll_loop`'s tight bound — the multiply-add count has not changed — and its intermediate tensors are $O(PC)$, not the $O(NAC)$ of Part 1's question 6, because `nonzero` selects the $P$ pairs before any arithmetic runs. What changes is that those $P$ products are now one call into `torch.bmm` instead of $P$ Python-level calls. Checked against `nll_loop` on the trained `p`, `M`, `Y` of the problem statement, the two agree to floating-point tolerance, and so do their gradients with respect to `p` and `M`. Timed over 30 repetitions after a few warm-up calls, `nll_loop` takes about 21 ms and `nll_vectorized` well under 1 ms — over a hundred times faster at this scale ($N = 400$, $A = 5$, $P = 998$), almost entirely the Python-loop overhead from Part 1's question 7, not a difference in algorithm.

### Bonus

- Time $O(L^2 d + Ld^2)$: splitting into $h$ heads of dimension $d / h$ does not change the first term, since each head's $QK^\top$ and (weights) $\times V$ cost $O(L^2 \cdot d/h)$ and there are $h$ of them; the four projections ($Q$, $K$, $V$, output) each cost $O(Ld^2)$ regardless of $h$. Memory to run backward is dominated by $O(hL^2)$: the post-softmax attention weights of every head have to be kept for softmax's backward pass, and unlike the time cost, this grows linearly with $h$, not only with $d$. This is exactly the tensor that fused kernels such as FlashAttention avoid materializing, recomputing it during backward instead to bring memory down to $O(Ld)$.
- Time $O(B \cdot C_{in} \cdot C_{out} \cdot k^2 \cdot H_{out} \cdot W_{out})$: every one of the $B \cdot C_{out} \cdot H_{out} \cdot W_{out}$ output entries is a sum over $C_{in} \cdot k^2$ input entries.
- Time $O(Bnmp)$: $B$ independent $(n, m) @ (m, p)$ products, each $O(nmp)$. The output has shape $(B, n, p)$, so it occupies $O(Bnp)$ memory.

### Follow-ups

- Computing the full grid with `einsum('nc,acd->nad', p, M)` and masking afterwards is mathematically the same computation as Part 3's version, but back to $O(NAC)$ memory; the only difference is selecting the valid pairs before or after doing the arithmetic.
- `q.gather(1, lab.unsqueeze(1))` could equally be written `q[torch.arange(len(lab)), lab]`; the two are equivalent, and `gather` generalizes more easily to higher dimensions.
- Training on mini-batches instead of the full `X` scales every step's cost down with the batch size, but it does not change the complexity conclusions of Part 1's questions 5 and 6 — it only spreads the same constants over more steps.
- With larger $C$, the `1e-8` floor inside `nll_loop` and `nll_vectorized` can stop being enough; working in log-probabilities throughout (log-softmax, `logsumexp`) rather than exponentiating and then taking a log again is the more robust choice.

```python
import statistics as st

# Part 2: the learned confusion matrices recover the reliability ranking, without ever seeing y
with torch.no_grad():
    r_hat = crowd.reliability()
print("estimated reliability:", [round(v, 2) for v in r_hat.tolist()])
assert least_reliable(r_hat) == RELIABILITY.index(min(RELIABILITY))

for extra_seed in (1, 2, 3):                    # a few more runs, fewer epochs, same conclusion
    _, crowd_s, _, Y_s = run(seed=extra_seed, epochs=15)
    with torch.no_grad():
        r_s = crowd_s.reliability()
    assert least_reliable(r_s) == RELIABILITY.index(min(RELIABILITY))

# Part 3: same value and same gradient as the loop, on the trained model of the problem statement
with torch.no_grad():
    p, M = F.softmax(net(X), dim=-1), crowd()
    loop_val, vec_val = nll_loop(p, M, Y), nll_vectorized(p, M, Y)
    assert torch.allclose(loop_val, vec_val, atol=1e-4)

p_g = F.softmax(net(X), dim=-1).detach().requires_grad_(True)
M_g = crowd().detach().requires_grad_(True)
g_loop = torch.autograd.grad(nll_loop(p_g, M_g, Y), [p_g, M_g])
p_g2, M_g2 = p_g.detach().requires_grad_(True), M_g.detach().requires_grad_(True)
g_vec = torch.autograd.grad(nll_vectorized(p_g2, M_g2, Y), [p_g2, M_g2])
assert torch.allclose(g_loop[0], g_vec[0], atol=1e-5) and torch.allclose(g_loop[1], g_vec[1], atol=1e-5)

torch.set_num_threads(1)                        # a stable reading: thread-pool scheduling otherwise adds noise
for _ in range(5):                              # warm-up before timing
    nll_loop(p, M, Y)
    nll_vectorized(p, M, Y)
reps = 30
t0 = time.time()
for _ in range(reps):
    nll_loop(p, M, Y)
t_loop = (time.time() - t0) / reps
t0 = time.time()
for _ in range(reps):
    nll_vectorized(p, M, Y)
t_vec = (time.time() - t0) / reps
print(f"nll_loop {t_loop * 1000:.2f} ms   nll_vectorized {t_vec * 1000:.2f} ms   speedup {t_loop / t_vec:.0f}x")
assert t_vec < t_loop / 10                      # vectorized version is at least an order of magnitude faster
```
