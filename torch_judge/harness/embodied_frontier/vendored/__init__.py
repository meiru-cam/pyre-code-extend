"""Ground-truth oracles transcribed from upstream frontier VLA and world-model code.

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
  * openvla_oft_parallel_decoding.py   — MIT License, (c) 2025 Moo Jin Kim, Chelsea Finn, Percy Liang (openvla-oft);
                                         Apache License 2.0, (c) 2022 EleutherAI and the HuggingFace Inc. team
                                         (transformers-openvla-oft)
  * lapa_univla_vq.py                  — MIT License (LAPA nsvq.py); Apache License 2.0 (UniVLA)
  * lerobot_knowledge_insulation.py    — Apache License 2.0, (c) 2025-2026 Physical Intelligence and the HuggingFace
                                         Inc. team (LeRobot), plus transformers eager attention (Apache License 2.0)
  * recap_paper.py                     — no code copied; transcribed from the formulas of pi*0.6 (arXiv 2511.14759)
  * cosmos_tokenizer_quantizers.py     — Apache License 2.0, NVIDIA CORPORATION & AFFILIATES (Cosmos-Tokenizer)
  * diffusion_forcing_df_base.py       — MIT License, (c) 2024 Boyuan Chen (diffusion-forcing);
                                         MIT License, (c) 2024 Etched & Decart (open-oasis)
  * dreamer4_shortcut_paper.py         — no code copied from Dreamer 4; formulas from arXiv 2509.24527, bootstrap target
                                         following kvfrans/shortcut-models (MIT License, (c) 2024 Kevin Frans)
  * self_forcing_training.py           — Apache License 2.0, the Self-Forcing authors (guandeh17/Self-Forcing)
"""
