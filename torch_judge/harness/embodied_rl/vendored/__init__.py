"""Ground-truth oracles transcribed from upstream robot RL environment code.

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
  * gymnasium_core.py              — MIT License, (c) 2016 OpenAI, (c) 2022 Farama Foundation (Gymnasium);
                                     step-flag and sparse-reward logic follow ManiSkill, Apache License 2.0
  * openpi_svr_action_adapter.py   — Apache License 2.0, Physical Intelligence (openpi);
                                     MIT License, (c) 2025 PRIME-RL (SimpleVLA-RL)
  * rlinf_maniskill_chunk_step.py  — Apache License 2.0, the RLinf authors and the ManiSkill authors
  * gymnasium_sync_vector_env.py   — MIT License, (c) 2016 OpenAI, (c) 2022 Farama Foundation (Gymnasium)
  * maniskill_vector_env.py        — Apache License 2.0, the ManiSkill authors
  * maniskill_ppo_gae.py           — Apache License 2.0, the ManiSkill authors and the RLinf authors
  * simplevla_rl_core_algos.py     — MIT License, (c) 2025 PRIME-RL (SimpleVLA-RL)
  * rlinf_flow_sde.py              — Apache License 2.0, (c) 2025-2026 The RLinf Authors
"""
