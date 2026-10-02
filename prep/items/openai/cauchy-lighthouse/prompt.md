A lighthouse stands in the sea at perpendicular distance $d > 0$ from a long straight shore. Put a coordinate on the shore and let $x_0$ be the coordinate of the shore point nearest the lighthouse. The lamp sends out a narrow beam; its direction is described by the angle $\Theta$ between the beam and the perpendicular from the lighthouse to the shore, and $\Theta$ is uniform on $(-\pi/2,\; \pi/2)$, so the beam always reaches the shore. Detectors along the shore record the coordinate $X$ of the point where it lands. Successive flashes are independent, with the same $d$ and $x_0$ every time.

Example with $d = 2$ and $x_0 = -3$, in kilometres:

```text
theta =  0       beam perpendicular to the shore   x = -3 + 2 *  0.00  =  -3.00
theta =  pi/4    tan(theta) =  1.00                x = -3 + 2 *  1.00  =  -1.00
theta = -1.45    tan(theta) = -8.24                x = -3 + 2 * -8.24  = -19.48
theta =  1.52    tan(theta) = 19.67                x = -3 + 2 * 19.67  =  36.34
```

Each of the four parts asks for a derivation. There is nothing to implement.

### Part 1 — Where one flash lands

Derive the distribution function $F(x) = P(X \le x)$ and the density $f$ of $X$. Write $X$ as $x_0 + dZ$ for a random variable $Z$ whose law depends on neither $x_0$ nor $d$, give the density of $Z$, and name the family. Where are the median and the quartiles of $X$, and how fast does the tail of $F$ decay?

### Part 2 — Averaging the flashes

Does $E[X]$ exist? Then let $X_1, \dots, X_n$ be the positions of $n$ flashes and find the exact distribution of the sample mean $\bar X_n = \frac1n \sum_{i=1}^{n} X_i$ for every $n$. What does the answer say about using $\bar X_n$ to locate the lighthouse, and which step of the usual "averaging shrinks the error by $\sqrt n$" argument fails?

### Part 3 — Estimators that do converge

For this part $d$ is known, $x_0$ is not, and the $n$ recorded positions are the only data; the angles are not observed. Call a sequence of estimators $T_n = T_n(X_1, \dots, X_n)$ *consistent* if $T_n \to x_0$ in probability, and say it has *asymptotic variance* $v$ if $\sqrt n\;(T_n - x_0)$ converges in distribution to $N(0,\; v)$; among consistent estimators, smaller $v$ is better. The benchmark is the *Fisher information* $I = E\bigl[(\partial_\theta \ln f(X; \theta))^{2}\bigr]$ at $\theta = x_0$, where $f(\cdot\;; \theta)$ is the density of Part 1 with $x_0$ replaced by $\theta$; an estimator with $v = 1/I$ is called *efficient*.

Give two consistent estimators of $x_0$, prove that each one is consistent, compute $I$ and the asymptotic variance of each estimator, and say which one is efficient and by how much the other misses.

### Part 4 — What a simulation would have to show

Suppose the three conclusions above are now checked numerically, by drawing angles and forming the landing positions. For each of them — the law of $X$, the law of $\bar X_n$, the rate at which the estimators of Part 3 converge — name the quantity you would measure and the number your derivation predicts for it. The mean-squared error $\frac1R \sum_{r=1}^{R} (T^{(r)} - x_0)^{2}$ over $R$ repetitions is a poor instrument here; say why, and give a measure of error that behaves.
