`X_train` is a float array of shape `(2000, 60)`; its rows are training points in $\mathbb{R}^{60}$. `X_test` and `y_test`, of shape `(1000, 60)` and `(1000,)`, are a held-out test set whose true class is in `{0, 1, 2}`. The training points have no single trusted label. Instead, `annotations` is an integer array of shape `(2000, 6)`: `annotations[i, j]` is the class that annotator `j` assigned to sample `i`, or `-1` if annotator `j` did not label sample `i`. Every row has at least two entries that are not `-1`. A minority of the 6 annotators are unreliable on the samples they labeled: some label close to randomly, and some are systematically biased toward one class.

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

N_TRAIN, N_TEST, N_FEATURES, N_CLASSES, N_ANNOTATORS = 2000, 1000, 60, 3, 6


def make_dataset(seed):
    """Returns (X_train, y_train_true, X_test, y_test, annotations), built deterministically from seed.
    y_train_true, shape (2000,), is the ground-truth class of every training point: use it only to
    check your answer to Part 1, never to build a training label. annotations, shape (2000, 6), has
    entries in {-1, 0, 1, 2}."""
    rng = np.random.default_rng(seed)
    X, y = make_classification(n_samples=N_TRAIN + N_TEST, n_features=N_FEATURES, n_informative=40,
                                n_redundant=10, n_classes=N_CLASSES, n_clusters_per_class=2,
                                class_sep=1.6, flip_y=0.01, random_state=seed)
    X_train, X_test = X[:N_TRAIN], X[N_TRAIN:]
    y_train, y_test = y[:N_TRAIN], y[N_TRAIN:]

    annotations = np.full((N_TRAIN, N_ANNOTATORS), -1)
    for i in range(N_TRAIN):
        k = rng.integers(2, 5)                                    # 2 to 4 annotators label this sample
        for j in rng.choice(N_ANNOTATORS, size=int(k), replace=False):
            if j == 4:                                             # NOTE: ignores X_train and y_train entirely
                annotations[i, j] = rng.integers(0, N_CLASSES)
            elif j == 5:                                           # NOTE: systematically biased, not noisy
                annotations[i, j] = 0
            elif rng.random() < 0.92:
                annotations[i, j] = y_train[i]
            else:
                others = [c for c in range(N_CLASSES) if c != y_train[i]]
                annotations[i, j] = rng.choice(others)
    return X_train, y_train, X_test, y_test, annotations
```

Build a pipeline that turns `annotations` into training labels and a classifier, in three parts.

### Part 1 — Aggregating votes and scoring annotators

**(a)** Implement `aggregate_labels(annotations, n_classes, weights=None)`. For each sample, sum the weight of every annotator who labeled it into that annotator's chosen class (weight `1.0` for every annotator when `weights` is `None`), and predict the class with the highest total. Break ties by the smaller class index. A sample nobody labeled — or one where every annotator who labeled it has weight `0` — has no prediction.

```py
def aggregate_labels(annotations: np.ndarray, n_classes: int,
                      weights: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """annotations: (n, A) int, -1 for missing. weights: (A,) or None (all-ones).
    Returns (labels, has_label): labels[i] is the predicted class (arbitrary if has_label[i] is False),
    has_label[i] is False iff sample i got no label with positive total weight."""
```

**(b)** Implement `annotator_reliability(annotations, n_classes)`: for annotator `j`, the *reliability score* is the fraction of the samples it labeled on which its label agrees with `aggregate_labels` of every *other* annotator (unweighted). A sample fewer than one other annotator labeled does not count towards `j`'s score.

```py
def annotator_reliability(annotations: np.ndarray, n_classes: int) -> np.ndarray:
    """Returns r of shape (A,), each entry in [0, 1]."""
```

**(c)** Implement `flag_bad_annotators(r)`: flag annotator `j` when its score is more than one standard deviation below the mean of all `A` scores.

```py
def flag_bad_annotators(r: np.ndarray) -> np.ndarray:
    """r: (A,). Returns a boolean array of shape (A,), True where the annotator is flagged."""
```

Example, `n_classes = 2`, 7 samples and 4 annotators (`-1` = not labeled):

```text
ann0 ann1 ann2 ann3
 0    0    0    1
 1    1    1    1
 0    0    1    1
 0    0    0    1
 1    0    0    1
 1    1    1    1
 0   -1    1    1
```

`annotator_reliability` returns approximately `[0.57, 0.67, 0.57, 0.29]`, and `flag_bad_annotators` flags only annotator 3. `aggregate_labels` on the full matrix returns `[0, 1, 0, 0, 0, 1, 1]`.

### Part 2 — Filter and retrain

Drop the columns of the annotators flagged in Part 1, aggregate the remaining columns with `aggregate_labels`, and discard the rows left with no label. Train `sklearn.linear_model.LogisticRegression(max_iter=2000)` on the surviving `(X_train, label)` pairs and score it on `(X_test, y_test)`. Compare against a baseline trained the same way on the labels aggregated from *all* six annotators, unfiltered.

```py
def fit_and_score(X_train: np.ndarray, labels: np.ndarray, has_label: np.ndarray,
                   X_test: np.ndarray, y_test: np.ndarray) -> float:
    """Fits LogisticRegression(max_iter=2000) on X_train[has_label], labels[has_label].
    Returns its accuracy on (X_test, y_test)."""
```

On the matrix above, dropping annotator 3 and re-aggregating the remaining three columns changes the label of the last sample from `1` to `0`.

### Part 3 — Weighted voting without dropping samples

Dropping an annotator also drops every sample whose winning label depended on that annotator's vote — with `d = 60` features that can leave too little training data. Reuse the reliability scores of Part 1 instead: derive and implement a weight for each annotator so that `aggregate_labels(annotations, n_classes, weights=w)` down-weights an unreliable annotator's vote rather than discarding their samples, and compare its test accuracy and its number of labeled training samples against Part 2.

```py
def reliability_weights(r: np.ndarray, n_classes: int) -> np.ndarray:
    """r: (A,), values in [0, 1]. Returns weights of shape (A,), each >= 0."""
```

On the matrix above, `reliability_weights` returns approximately `[0.29, 0.69, 0.29, 0.0]`, and `aggregate_labels` with these weights on the full four-column matrix again predicts `0` for the last sample — without dropping annotator 3's column.
