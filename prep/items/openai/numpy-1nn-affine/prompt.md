`X_train` is a float array of shape `(n, d)` with $n \ge 1$. Its rows $x_0, \dots, x_{n-1} \in \mathbb{R}^d$ are the training points. `y_train` is an integer array of shape `(n,)`, and `y_train[i]` is the class label of $x_i$, one of the integers $0, 1, \dots, c - 1$. `X_query` is a float array of shape `(m, d)` whose rows $q_0, \dots, q_{m-1}$ are the points to classify.

*1-nearest-neighbour (1-NN) classification* gives a query $q$ the label of the training point closest to it. Closeness is measured by the *squared Euclidean distance*

$$\lVert q - x_i \rVert^2 = \sum_{k=0}^{d-1} (q_k - x_{ik})^2 ,$$

and when several training points are equally close, the one with the smallest index wins:

$$\mathrm{nn}(q) = \text{the smallest } i \text{ such that } \lVert q - x_i \rVert^2 = \min_k \lVert q - x_k \rVert^2 .$$

The predicted label of $q$ is `y_train[nn(q)]`. Take $1 \le n, m \le 2000$ and $1 \le d \le 50$ throughout. Implement the following two parts using NumPy only, without calling `scipy.spatial`, `sklearn`, or any other external distance or machine-learning library.

### Part 1 — 1-NN without Python loops

Return the predicted labels of all $m$ queries. The code must not iterate in Python over training points, queries or coordinates: no `for` or `while`, no comprehension, no `map`, no `np.vectorize`. Every step is an operation on whole arrays. State the shape of every intermediate array.

```py
def predict_1nn(X_train: np.ndarray, y_train: np.ndarray, X_query: np.ndarray) -> np.ndarray:
    """Shapes: X_train (n, d) float, y_train (n,) int, X_query (m, d) float.
    Returns the predicted labels, shape (m,)."""
```

Example with $n = 4$, $d = 2$, $m = 3$:

```text
X_train = [[0, 0],        y_train = [2, 0, 0, 1]        X_query = [[1, 0],
           [2, 0],                                                 [2, 1],
           [0, 2],                                                 [2, 3]]
           [3, 3]]
```

- query: $q_0 = (1, 0)$ · sq. distance to $x_0$: 1 · to $x_1$: 1 · to $x_2$: 5 · to $x_3$: 13 · $\mathrm{nn}(q)$: 0 (tie between 0 and 1) · predicted label: `y_train[0]` = 2
- query: $q_1 = (2, 1)$ · sq. distance to $x_0$: 5 · to $x_1$: 1 · to $x_2$: 5 · to $x_3$: 5 · $\mathrm{nn}(q)$: 1 · predicted label: `y_train[1]` = 0
- query: $q_2 = (2, 3)$ · sq. distance to $x_0$: 13 · to $x_1$: 9 · to $x_2$: 5 · to $x_3$: 1 · $\mathrm{nn}(q)$: 3 · predicted label: `y_train[3]` = 1

The function returns `[2, 0, 1]`.

### Part 2 — The same prediction as an affine layer followed by an activation

An *affine layer* (also called a fully connected or linear layer, `torch.nn.Linear` in PyTorch) maps an input vector $x \in \mathbb{R}^d$ to $Wx + b$. The *weight matrix* $W$ and the *bias vector* $b$ do not depend on $x$. An *activation function* is a fixed non-linear function applied to the output of a layer. Here it is the *softmax*, which maps $z \in \mathbb{R}^n$ to a vector of $n$ probabilities:

$$\mathrm{softmax}(z)_i = \frac{e^{z_i}}{\sum_{k} e^{z_k}} .$$

The network of this part sends a query through one affine layer with $n$ outputs, applies the softmax, takes the index of the largest probability (the smallest such index if there are several, which is what `np.argmax` returns) and looks this index up in `y_train`.

**(a)** For a single query $q \in \mathbb{R}^d$ written as a column vector, find $W_1 \in \mathbb{R}^{n \times d}$ and $b_1 \in \mathbb{R}^n$, computed from the training set alone, such that for every $q$

$$\arg\max_i \mathrm{softmax}(W_1 q + b_1)_i = \mathrm{nn}(q) ,$$

including the rule for ties. Prove the equality.

**(b)** Implement the forward pass for the whole batch `X_query`, whose rows are the queries: `logits = X_query @ W + b` with `W` of shape `(d, n)` and `b` of shape `(n,)`, followed by the softmax of each row. `X_query` may be used in this one expression only. Say how `W` and `b` are obtained from $W_1$ and $b_1$.

```py
def affine_layer(X_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Builds the layer from X_train of shape (n, d). Returns W of shape (d, n) and b of shape (n,)."""

def predict_1nn_affine(X_train: np.ndarray, y_train: np.ndarray,
                       X_query: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Forward pass. Returns (probs, labels): probs[j] = softmax(logits[j]), shape (m, n),
    and labels[j] = y_train[argmax(probs[j])], shape (m,)."""
```

For the data of Part 1, `probs` has shape `(3, 4)`, each of its rows sums to 1, and `labels` is again `[2, 0, 1]`. For every input, `labels` must equal the output of `predict_1nn`.
