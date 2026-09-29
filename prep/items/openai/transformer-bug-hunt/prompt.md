The file `tiny_gpt.py` below implements a small decoder-only Transformer in PyTorch and trains it to reverse a string of digits.

A training sequence has 13 tokens: six random digits (tokens `0` to `9`), the separator token `10`, and the same six digits in reverse order.

```text
position   0  1  2  3  4  5   6   7  8  9 10 11 12
token      7  0  2  2  5  8  10   8  5  2  2  0  7
```

The model receives the first 12 tokens and outputs, at every position $t$, the logits for the token at position $t + 1$. Self-attention is *causal*: position $t$ may attend to positions $\le t$ only. The loss is the cross-entropy over the last six targets, which are the reversed digits. `run_tests` checks three things:

- changing the last input token leaves the logits at all earlier positions unchanged;
- the final training loss is below 0.05;
- given the six digits and the separator, greedy decoding with `generate` completes more than 99% of 500 new sequences without a single wrong token.

### Part 1 — Find and fix four bugs

The file contains four bugs: one in the set-up of the positional embedding, two in `SelfAttention`, and one in the training loop. Three of them are a wrong line and one is a missing line. For each bug, give its location, explain what it does to training or to the output and why, and fix it. With all four fixed, `python tiny_gpt.py` reaches a loss below 0.05 within its 400 steps (a few seconds on a CPU) and prints `all tests passed`. The data, the tests and the hyperparameters are correct and need no change.

```py
import torch
import torch.nn as nn
import torch.nn.functional as F

N_DIGITS, SEP = 6, 10                       # tokens 0..9 are digits, token 10 is the separator
VOCAB, MAX_LEN = 11, 2 * N_DIGITS + 1


class SelfAttention(nn.Module):
    def __init__(self, d_model, n_head, max_len):
        super().__init__()
        self.n_head, self.d_head = n_head, d_model // n_head
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(self.d_head, d_model)
        # visible[i, j] is True when the query at position i may attend to the key at position j
        self.register_buffer("visible", torch.ones(max_len, max_len).tril().bool())

    def split_heads(self, x):               # (B, T, d_model) -> (B, n_head, T, d_head)
        return x.unflatten(-1, (self.n_head, self.d_head)).transpose(1, 2)

    def forward(self, x):
        T = x.shape[1]
        q, k, v = self.split_heads(self.w_q(x)), self.split_heads(self.w_k(x)), self.split_heads(self.w_v(x))
        scores = q @ k.transpose(-1, -2) / self.d_head ** 0.5           # (B, n_head, T, T)
        scores = scores.masked_fill(~self.visible[:T, :T], -1e-9)
        y = scores.softmax(dim=-1) @ v                                  # (B, n_head, T, d_head)
        return self.w_o(y.transpose(1, 2).flatten(-2))


class Block(nn.Module):
    def __init__(self, d_model, n_head, max_len):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.attn = SelfAttention(d_model, n_head, max_len)
        self.mlp = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class TinyGPT(nn.Module):
    def __init__(self, vocab=VOCAB, d_model=64, n_head=4, n_layer=1, max_len=MAX_LEN):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.blocks = nn.ModuleList([Block(d_model, n_head, max_len) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab)

    def forward(self, idx):                 # idx: (B, T) token ids -> logits (B, T, vocab)
        T = idx.shape[1]
        x = self.tok_emb(idx) + self.pos_emb[:, :T]
        for block in self.blocks:
            x = block(x)
        return self.head(self.ln_f(x))


def make_batch(batch_size, rng):            # each row: N_DIGITS random digits, SEP, the same digits reversed
    digits = torch.randint(0, 10, (batch_size, N_DIGITS), generator=rng)
    sep = torch.full((batch_size, 1), SEP)
    return torch.cat([digits, sep, digits.flip(1)], dim=1)


def train(model, steps=400, batch_size=64, lr=3e-3):
    rng = torch.Generator().manual_seed(0)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for step in range(steps):
        seq = make_batch(batch_size, rng)
        logits = model(seq[:, :-1])                                     # logits[:, t] predicts seq[:, t + 1]
        answer_logits, answer = logits[:, -N_DIGITS:], seq[:, -N_DIGITS:]   # only the reversed digits are scored
        loss = F.cross_entropy(answer_logits.reshape(-1, VOCAB), answer.reshape(-1))
        opt.zero_grad()
        opt.step()
        if step % 100 == 0 or step == steps - 1:
            print(f"step {step:3d}  loss {loss.item():.4f}")
    return loss.item()


@torch.no_grad()
def generate(model, prompt, n_new):         # greedy decoding: append the most likely next token, n_new times
    seq = prompt
    for _ in range(n_new):
        next_token = model(seq)[:, -1].argmax(dim=-1, keepdim=True)
        seq = torch.cat([seq, next_token], dim=1)
    return seq


@torch.no_grad()
def run_tests(model, final_loss):
    seq = make_batch(500, torch.Generator().manual_seed(1))
    a, b = seq[:, :-1].clone(), seq[:, :-1].clone()
    b[:, -1] = (b[:, -1] + 1) % 10                                      # a and b differ in the last token only
    early_a, early_b = model(a)[:, :-1], model(b)[:, :-1]               # logits at all positions before the change
    assert torch.allclose(early_a, early_b, atol=1e-5), "test 1: a later token changed earlier logits"
    assert final_loss < 0.05, "test 2: the loss did not converge"
    completed = generate(model, seq[:, :N_DIGITS + 1], N_DIGITS)
    accuracy = (completed == seq).all(dim=1).float().mean().item()
    assert accuracy > 0.99, f"test 3: only {accuracy:.1%} of the generated answers are right"


if __name__ == "__main__":
    torch.manual_seed(0)
    model = TinyGPT()
    final_loss = train(model)
    run_tests(model, final_loss)
    print("all tests passed")
```

### Part 2 — Odd/even classifier

Turn the fixed model into a classifier that labels a number as odd or even. The input `idx` of shape `(B, T)` holds the decimal digits of each number, most significant digit first, without a separator; leading zeros are allowed. For example, `7 0 2 2 5 8` is even and `8 5 2 2 0 7` is odd. Replace the vocabulary head with a two-class head. Before that head, average the final hidden states (the output of `ln_f`, of shape `(B, T, d_model)`) over the $T$ positions, which is called *mean pooling*. Change the loss and the prediction to match. Train on random six-digit numbers until the classifier labels 1000 new numbers without an error.

```py
class ParityClassifier(nn.Module):
    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """idx: (B, T) digit tokens. Returns logits of shape (B, 2): class 0 = even, class 1 = odd."""
```

### Part 3 — KV cache

`generate` calls the model once per new token, and every call recomputes the keys and values of all earlier positions. Under causal attention they do not depend on later tokens, so they never change. A *KV cache* stores, for every layer, the keys and values of the positions processed so far, so that a call only has to process the new tokens. The class below is given, and its method `update` is missing.

```py
class KVCache:
    """Keys and values of all positions processed so far, one entry per layer."""

    def __init__(self, n_layer: int):
        self.k = [None] * n_layer       # per layer: (B, n_head, T_past, d_head), or None while empty
        self.v = [None] * n_layer

    @property
    def length(self) -> int:            # T_past
        return 0 if self.k[0] is None else self.k[0].shape[2]

    def update(self, layer: int, k: torch.Tensor, v: torch.Tensor):
        """k, v: (B, n_head, T_new, d_head), the new positions of this layer. Appends them to the cache
        and returns the keys and values of all T_past + T_new positions."""
        raise NotImplementedError
```

Implement `update`, and add an optional argument `cache` to `TinyGPT.forward`, `Block.forward` and `SelfAttention.forward`. A call `model(idx, cache)` receives only the new tokens `idx` of shape `(B, T_new)`, appends their keys and values to the cache, and returns the logits of these `T_new` positions. A sequence is handed over as chunks of arbitrary lengths, in order, one call per chunk, and a call processes its own `T_new` positions only: the keys and values of the earlier positions are read from the cache, never recomputed. For any batch `seq` from `make_batch`, the logits of the chunks must equal the matching slices of a forward pass over the whole sequence:

```py
cache = KVCache(n_layer=len(model.blocks))
chunks = [(0, 3), (3, 4), (4, 9), (9, 11), (11, 13)]        # chunk lengths 3, 1, 5, 2, 2
out = [model(seq[:, a:b], cache) for a, b in chunks]
assert torch.allclose(torch.cat(out, dim=1), model(seq), atol=1e-5)
```

Then give `generate` a flag `use_cache`: the first call processes the prompt, and every later call processes only the newest token.
