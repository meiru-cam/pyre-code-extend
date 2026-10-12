"""Ground-truth oracles transcribed from upstream robot-learning code.

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
  * openvla_action_tokenizer.py        — MIT License, the OpenVLA authors
  * fast_action_tokenizer.py           — Apache License 2.0, Physical Intelligence
  * lerobot_act_temporal_ensembler.py  — Apache License 2.0, the HuggingFace Team
  * lerobot_flow_matching.py           — Apache License 2.0, the HuggingFace Team
  * lerobot_rtc.py                     — Apache License 2.0, the HuggingFace Team
"""
