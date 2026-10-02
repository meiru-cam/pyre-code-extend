Two points worth confirming with the interviewer: whether an annotator's reliability should be scored against the majority of every other annotator (assumed here) or against a trusted subset, and whether a flagged annotator's labels should be dropped by default or down-weighted by default.

### Part 1

**(a)** A vote is a sum of weights per class, one addend per annotator who labeled the sample; the predicted class is the one with the largest sum, and `np.argmax` already breaks ties by the smallest index. `has_label` catches the two ways a sample can end up with no vote: nobody labeled it, or every annotator who did has weight `0`.

```python
def aggregate_labels(annotations, n_classes, weights=None):
    n, A = annotations.shape
    if weights is None:
        weights = np.ones(A)
    scores = np.zeros((n, n_classes))
    for j in range(A):
        labeled = annotations[:, j] != -1
        scores[labeled, annotations[labeled, j]] += weights[j]
    has_label = scores.sum(axis=1) > 0
    labels = np.argmax(scores, axis=1)               # NOTE: argmax keeps the FIRST max -> ties go to class 0
    return labels, has_label
```

**(b)** An annotator is compared against the majority of the *others*, never one that includes its own vote — with as few as two voters on a sample, folding an annotator's vote into its own reference majority would let it "agree with itself" and inflate the score.

```python
def annotator_reliability(annotations, n_classes):
    n, A = annotations.shape
    r = np.zeros(A)
    for j in range(A):
        peers = np.delete(annotations, j, axis=1)
        majority, has_majority = aggregate_labels(peers, n_classes)
        labeled_by_j = annotations[:, j] != -1
        mask = labeled_by_j & has_majority
        r[j] = np.mean(annotations[mask, j] == majority[mask]) if mask.any() else 0.0
    return r
```

**(c)** A fixed cutoff such as `0.5` would need retuning for a different number of classes; scoring against the annotator pool's own mean and spread does not.

```python
def flag_bad_annotators(r):
    return r < r.mean() - r.std()
```

At `seed=7`, `annotator_reliability` returns approximately `[0.67, 0.66, 0.66, 0.67, 0.35, 0.40]` and `flag_bad_annotators` flags exactly annotators 4 and 5 — the two built into `make_dataset` above. This holds for every seed checked below.

### Part 2

```python
def fit_and_score(X_train, labels, has_label, X_test, y_test):
    clf = LogisticRegression(max_iter=2000)
    clf.fit(X_train[has_label], labels[has_label])
    return accuracy_score(y_test, clf.predict(X_test))
```

At `seed=7`, the baseline (majority vote over all six annotators) reaches `0.72` test accuracy; after dropping annotators 4 and 5 and re-aggregating, the filtered classifier reaches `0.86`, trained on 1954 of the 2000 samples — the other 46 lost every one of their annotators once the flagged columns were removed. Every one of the annotators dropped by that pruning genuinely was one of the two implanted as unreliable, so no informative vote was thrown away with them; the improvement comes only from removing noise, never from removing signal.

### Part 3

Model annotator $j$ as a noisy channel: given the true class $c$, it reports $c$ with probability $r_j$ and each other class with probability $(1 - r_j) / (K - 1)$, where $K$ is `n_classes` — the same one shared error rate for annotator $j$ that Part 1 estimates, standing in for a full confusion matrix. For a sample with voters $V_i$ and candidate true class $c$, the log-likelihood of the votes it received is

$$L_i(c) = \sum_{j \in V_i} \Bigl( \mathbb{1}[\ell_{ij} = c] \log r_j
         + \mathbb{1}[\ell_{ij} \ne c] \log \frac{1 - r_j}{K - 1} \Bigr) .$$

Writing $\mathbb{1}[\ell_{ij} \ne c] = 1 - \mathbb{1}[\ell_{ij} = c]$ splits this into a term that does not depend on $c$ and one that does:

$$L_i(c) = \sum_{j \in V_i} \log \frac{1 - r_j}{K - 1}
         + \sum_{j \in V_i} \mathbb{1}[\ell_{ij} = c] \Bigl( \log r_j - \log \frac{1 - r_j}{K - 1} \Bigr) .$$

The first sum is the same for every $c$, so it drops out of the arg max, leaving exactly the weighted vote of `aggregate_labels`, with

$$w_j = \log \frac{r_j (K - 1)}{1 - r_j} .$$

This is a single, non-iterative pass of the same idea behind the Dawid–Skene EM algorithm: it reuses the $r_j$ that Part 1 already estimated instead of alternating between confusion matrices and soft labels.

```python
def reliability_weights(r, n_classes):
    r = np.clip(r, 1e-2, 1 - 1e-2)                    # NOTE: avoids log(0) at r == 0 or r == 1
    w = np.log(r * (n_classes - 1) / (1 - r))
    return np.clip(w, 0.0, None)                      # NOTE: chance-level or worse -> weight 0, not a negative vote
```

At `seed=7` the weights are approximately `[1.42, 1.34, 1.38, 1.42, 0.09, 0.28]`: annotator 4 (the uniform guesser) is worth almost nothing, and annotator 5 (biased to class 0) still carries a small positive weight, since one fixed answer is occasionally right by chance. Aggregating with these weights keeps all 2000 samples labeled — no vote total is ever driven to exactly `0` here — and the resulting classifier reaches `0.88` test accuracy, slightly above the `0.86` of Part 2, without giving up any training data.

### Follow-ups

- Majority vote itself can fail on a single sample when most of the annotators who happened to label it are unreliable, independently of how well Part 1 scores the *annotators*; wider per-sample coverage makes this rarer but never rules it out.
- Running the E/M steps of Dawid–Skene to convergence, instead of stopping at this one-shot version, lets each annotator have a full confusion matrix (different error rates per true class) rather than a single shared $r_j$.
- The reliability score here treats annotators independently; two annotators who share the same bias reinforce rather than catch each other. A pairwise annotator-by-annotator agreement matrix (or Cohen's kappa between each pair) surfaces that kind of correlated error, which agreement-with-the-majority alone cannot.

```python
toy = np.array([
    [0, 0, 0, 1],
    [1, 1, 1, 1],
    [0, 0, 1, 1],
    [0, 0, 0, 1],
    [1, 0, 0, 1],
    [1, 1, 1, 1],
    [0, -1, 1, 1],
])
r_toy = annotator_reliability(toy, 2)
bad_toy = flag_bad_annotators(r_toy)
full_labels, _ = aggregate_labels(toy, 2)
filt_labels, _ = aggregate_labels(toy[:, ~bad_toy], 2)
w_toy = reliability_weights(r_toy, 2)
w_labels, _ = aggregate_labels(toy, 2, weights=w_toy)

assert np.round(r_toy, 2).tolist() == [0.57, 0.67, 0.57, 0.29]
assert bad_toy.tolist() == [False, False, False, True]
assert full_labels.tolist() == [0, 1, 0, 0, 0, 1, 1]         # at the last row, annotator 3 tips 1-1 into a 2-1 win
assert filt_labels.tolist() == [0, 1, 0, 0, 0, 1, 0]         # dropping annotator 3 flips it back to a 1-1 tie -> 0
assert np.round(w_toy, 2).tolist() == [0.29, 0.69, 0.29, 0.0]
assert w_labels.tolist() == filt_labels.tolist()             # weighting recovers the same fix, no rows dropped

for seed in range(10):
    X_train, y_train_true, X_test, y_test, annotations = make_dataset(seed=seed)
    r = annotator_reliability(annotations, N_CLASSES)
    bad = flag_bad_annotators(r)
    assert np.where(bad)[0].tolist() == [4, 5]                # the two implanted bad annotators, every seed

    base_labels, base_has = aggregate_labels(annotations, N_CLASSES)
    base_acc = fit_and_score(X_train, base_labels, base_has, X_test, y_test)

    filt_labels, filt_has = aggregate_labels(annotations[:, ~bad], N_CLASSES)
    filt_acc = fit_and_score(X_train, filt_labels, filt_has, X_test, y_test)

    w = reliability_weights(r, N_CLASSES)
    w_labels, w_has = aggregate_labels(annotations, N_CLASSES, weights=w)
    weighted_acc = fit_and_score(X_train, w_labels, w_has, X_test, y_test)

    assert filt_acc > base_acc                     # filtering beats the unfiltered baseline
    assert weighted_acc > base_acc                  # so does weighting
    assert filt_has.sum() < base_has.sum()          # filtering silently drops some samples
    assert w_has.sum() == base_has.sum() == N_TRAIN # weighting never drops a sample here
```
