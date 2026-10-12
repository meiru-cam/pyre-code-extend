"""Ground-truth oracles transcribed from upstream latent-reasoning code.

Each module here carries its upstream URL, pinned commit, symbol list and
license in its docstring, plus an explicit account of what was changed and why.
The arithmetic is transcribed line for line; only infrastructure that cannot run
in the offline grading sandbox is removed.

These exist so an exercise is not graded solely against an oracle written by the
same author as its reference solution. Each exercise's oracle test in
`tests/quality/` compares the reference solution against its upstream
counterpart directly.

No module in this package may import `torch_judge.tasks`.

Licenses of vendored material:
  * pause_tokens_paper.py        — no code copied; transcribed from the formulas of
                                   Goyal et al., "Think before you speak" (arXiv 2310.02226)
  * coconut_dataset.py           — MIT License, Meta Platforms, Inc. and affiliates
  * coconut_model.py             — MIT License, Meta Platforms, Inc. and affiliates
  * huginn_recurrent_depth.py    — Apache License 2.0, the recurrent-pretraining authors
  * huginn_exit_evaluators.py    — Apache License 2.0, the recurrent-pretraining authors
"""
