You receive the materials below at the start of a 4-hour window; you may not prepare on this specific task beforehand. Work happens in your own local environment or a hosted notebook service — either is fine, as long as you can run Python with NumPy and scikit-learn. Documentation is allowed and expected, but no AI coding assistant may be used, and every line of code and every conclusion must be your own. At the end of the 4 hours, give a 10-minute presentation — slides or a notebook walkthrough are both fine, as long as the results are shown clearly — followed by a 30-minute technical discussion of the design choices behind the experiment, which of the conclusions the data actually supports, and what you would change with more time.

*Symmetric label noise* at rate $p$ is a corruption process applied to training labels only: independently for each training example, with probability $p$ its label is replaced by one of the other classes, chosen uniformly at random, and with probability $1 - p$ it is left unchanged.

Research question: for a one-hidden-layer classifier trained on synthetic data, how does symmetric label noise in the training labels interact with the size of the training set and the network's hidden width, and how much of the resulting damage does early stopping against a small clean validation set recover?

### The data

```python
import numpy as np
from sklearn.datasets import make_classification

D, C = 20, 3
N_VAL, N_TEST = 200, 1000


def make_dataset(seed: int, n_train: int, noise_rate: float):
    rng = np.random.default_rng(seed)
    n_total = n_train + N_VAL + N_TEST
    X, y = make_classification(
        n_samples=n_total, n_features=D, n_informative=15, n_redundant=3,
        n_classes=C, n_clusters_per_class=2, class_sep=1.8, flip_y=0.0, random_state=seed,
    )
    X_train, X_val, X_test = X[:n_train], X[n_train:n_train + N_VAL], X[n_train + N_VAL:]
    y_train, y_val, y_test = y[:n_train], y[n_train:n_train + N_VAL], y[n_train + N_VAL:]

    y_train_noisy = y_train.copy()
    flip = rng.random(n_train) < noise_rate
    for i in np.where(flip)[0]:
        other_classes = [c for c in range(C) if c != y_train[i]]
        y_train_noisy[i] = rng.choice(other_classes)
    return X_train, y_train_noisy, y_train, X_val, y_val, X_test, y_test
```

`make_dataset` is deterministic given `seed`: it draws `n_train + 200 + 1000` points from one `make_classification` call (20 features — 15 informative, 3 redundant, 2 pure noise — 3 balanced classes, 2 clusters per class), splits them into a training set of size `n_train`, a validation set of 200 points, and a test set of 1000 points, and applies symmetric noise at `noise_rate` to the training labels only. `y_train_true`, the label before noise, is returned for diagnostics; it must never be used to fit or select a model. `X_val`, `y_val`, `X_test`, and `y_test` are always clean.

### The model and the experiment grid

The model is a classifier with one hidden layer: `D` inputs, a hidden layer of width $H$ with a ReLU nonlinearity, and a softmax output over the `C` classes, trained to minimize the cross-entropy of the (possibly noisy) training labels. The optimizer, the learning rate, the number of epochs, and any weight decay are your choice; state what you used and how you picked it.

Run every combination of:

- hidden width $H \in \{8, 128\}$;
- training-set size $n_{\text{train}} \in \{100, 400, 1600\}$;
- noise rate $p \in \{0.0, 0.2, 0.4\}$,

with at least 5 seeds per combination (the same seed drives both `make_dataset` and the model's weight initialization). For every seed, in addition to the test accuracy after the last training epoch, record validation accuracy after every epoch and the test accuracy at the epoch with the best validation accuracy — the accuracy early stopping against `X_val`, `y_val` would have delivered, without `X_test`, `y_test` ever being used to choose it.

### Questions to answer

- Within each hidden width, how does test accuracy depend on the noise rate and on the training-set size?
- Does the hidden width change how much noise hurts test accuracy, and does that change with training-set size?
- Does early stopping recover test accuracy lost to noise? Does the size of the recovery depend on the noise rate, the hidden width, or the training-set size — and does early stopping cost anything when there is no noise at all?
- Which of the trends behind your answers are well supported by 5 seeds, and which are not? State the margin you would want to see before trusting a comparison.

Hand in two tables — one of test accuracy (mean and spread across seeds) for the last-epoch models, one of early stopping's recovery (mean test accuracy at the best-validation epoch minus mean test accuracy at the last epoch, over the same seeds) — answers to the four questions that reference those tables, and the presentation.
