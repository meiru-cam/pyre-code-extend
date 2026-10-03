You have four hours. Design and run a small experiment that shows sample-wise double descent, explain why it happens, and propose and demonstrate a fix.

**Background.** Fix a model class and a model size — for a linear model, the number of features; for a network, its width or parameter count. Train it on $n$ i.i.d. samples drawn from a fixed distribution and measure its error on held-out data from the same distribution, its *test error*. In the classical regime, more training data makes test error decrease monotonically. A model large enough to *interpolate* its training set — to reach zero training error — can instead show *double descent*: as $n$ grows at fixed model size, test error decreases, rises to a peak near the *interpolation threshold* — the smallest $n$, at that model size, at which the model has just enough capacity to fit the training data exactly — and then decreases again once $n$ is well past the threshold. Varying $n$ at fixed model size, as here, is the *sample-wise* version of the phenomenon; varying model size at fixed $n$ instead is the *model-wise* version.

**Task.**

- Choose a model class, a way of generating labelled data from a known ground truth, and a definition of test error. Choose how you will vary $n$ at a fixed model size.
- Compute test error at enough values of $n$ — spanning well below the interpolation threshold to well above it — to trace out the double-descent curve, averaging over enough random repetitions of the training data that the peak is not an artefact of one unlucky draw.
- Explain, in terms of the model class you chose, why test error rises to a peak near the interpolation threshold and falls on both sides of it.
- Propose one remedy that removes or reduces the peak without materially changing the rest of the curve, and demonstrate it on the same experiment.

**Deliverables.** A notebook containing the experiment's code, its numerical results, and brief commentary; and a slide deck of at most eight slides for a live review, covering the question, your setup, the result, your explanation, the remedy, its limitations, and what you would try next.

**Constraints.** Four hours, start to finish. NumPy and a plotting library may be used, along with the standard library; nothing else is required at this scale, and no machine-learning framework is needed. Do not use an AI coding assistant — the experiment, the explanation and the slides must be your own work.

**Live review.** After you submit the notebook and slides, you present them and answer questions: why you made the choices you did, whether your explanation would still hold under a different data distribution or a different model class, and how you would extend the experiment given more time.
