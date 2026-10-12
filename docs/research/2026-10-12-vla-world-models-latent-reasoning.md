# Research: a VLA, World Models and Latent Reasoning path

Date: 2026-10-12
Status: Research input for curriculum design. No exercises were created.
Scope: three modules (Vision-Language-Action models, World Models, Latent Reasoning) for one new learning path, grounded in primary sources.

How this was researched: upstream repositories were shallow-cloned on 2026-10-12 and read directly; commit hashes below are the `HEAD` of those clones. Licenses were read from each repo's `LICENSE` file (and cross-checked with the GitHub API where noted). arXiv titles were checked against `arxiv.org/abs/<id>`. Hugging Face model-card licenses and revisions come from `https://huggingface.co/api/models/<id>`. Anything not verified this way is marked **Unverified**.

---

## 0. Pinned upstream sources

| Short name | Repo / artifact | Commit / revision (date) | License (verified from) |
|---|---|---|---|
| LeRobot | https://github.com/huggingface/lerobot | `75cd2ba0ce9ff47686bb2bd7b7f65840606c02ee` (2026-10-12) | Apache-2.0 (`LICENSE`) |
| openpi | https://github.com/Physical-Intelligence/openpi | `215abfb217dbac7d5f1273282331b9b1866c0479` (2026-08-24) | Apache-2.0 (`LICENSE`); repo also ships `LICENSE_GEMMA.txt` |
| FAST tokenizer | https://huggingface.co/physical-intelligence/fast | revision `ec4d7aa71691cac0b8bed6942be45684db2110f4` | apache-2.0 (model card metadata) |
| OpenVLA | https://github.com/openvla/openvla | `c8f03f48af692657d3060c19588038c7220e9af9` (2025-03-23) | MIT (`LICENSE`) |
| RT-1 | https://github.com/google-research/robotics_transformer | `4569641b8111f3f402c32d8e24becd2a6e952ecc` (2023-02-10) | Apache-2.0 (`LICENSE`); TensorFlow code |
| ACT (original) | https://github.com/tonyzhaozh/act | `742c753c0d4a5d87076c8f69e5628c79a8cc5488` (2024-01-28) | MIT (`LICENSE`) |
| Diffusion Policy (original) | https://github.com/real-stanford/diffusion_policy | `5ba07ac6661db573af695b419a7947ecb704690f` (2024-12-24) | MIT (`LICENSE`) |
| RTC (original) | https://github.com/Physical-Intelligence/real-time-chunking-kinetix | **commit not pinned** (not cloned) | MIT (GitHub API only) |
| DreamerV3 | https://github.com/danijar/dreamerv3 | `e01491fad6434b2245a3b8ca201dd7faedcc458c` (2026-10-03) | MIT (`LICENSE`); JAX code |
| DreamerV3 PyTorch port | https://github.com/NM512/dreamerv3-torch | `6ef8646d807cd10ce0c88e10a7e943211e7fc44c` (2026-03-08) | MIT (`LICENSE`) |
| V-JEPA 2 | https://github.com/facebookresearch/vjepa2 | `204698b45b3712590f06245fbfba32d3be539812` (2026-03-23) | MIT, with three named files under Apache-2.0 (`LICENSE`, README "License") |
| V-JEPA (v1) | https://github.com/facebookresearch/jepa | `51c59d518fc63c08464af6de585f78ac0c7ed4d5` (2025-02-27) | **CC BY-NC 4.0** (`LICENSE`) |
| I-JEPA | https://github.com/facebookresearch/ijepa | `52c1ae95d05f743e000e8f10a1f3a79b10cff048` (2023-06-13) | **CC BY-NC 4.0** (`LICENSE`) |
| open-genie (unofficial Genie) | https://github.com/myscience/open-genie | `732b9f9b746f18fff1a0fb22f83638224f2f7cc6` (2024-08-21) | MIT (`LICENSE` via GitHub API) |
| DIAMOND | https://github.com/eloialonso/diamond | `5bcd1599755b4f2fae8e5e079e02f0728e174965` (2024-12-06) | MIT (`LICENSE`) |
| IRIS | https://github.com/eloialonso/iris | `24326aaaa283c527f42b89b44cfdecf2665a7a16` (2024-10-14) | **GPL-3.0** (`LICENSE`) |
| Coconut | https://github.com/facebookresearch/coconut | `27273cb8cca4bb763c041a63b036d0c3b7cbbb48` (2025-08-12) | MIT (`LICENSE`) |
| Recurrent depth (Huginn) | https://github.com/seal-rg/recurrent-pretraining | `1ea7220ec7eb42d13e89db0663df254d0bcdc28e` (2025-12-29) | Apache-2.0 (`LICENSE`) |

Papers (titles verified on arXiv): RT-1 [2212.06817](https://arxiv.org/abs/2212.06817), RT-2 [2307.15818](https://arxiv.org/abs/2307.15818), OpenVLA [2406.09246](https://arxiv.org/abs/2406.09246), FAST [2501.09747](https://arxiv.org/abs/2501.09747), ACT [2304.13705](https://arxiv.org/abs/2304.13705), Diffusion Policy [2303.04137](https://arxiv.org/abs/2303.04137), pi0 [2410.24164](https://arxiv.org/abs/2410.24164), SmolVLA [2506.01844](https://arxiv.org/abs/2506.01844), GR00T N1 [2503.14734](https://arxiv.org/abs/2503.14734), VQ-BeT [2403.03181](https://arxiv.org/abs/2403.03181), RTC [2506.07339](https://arxiv.org/abs/2506.07339), DreamerV3 [2301.04104](https://arxiv.org/abs/2301.04104), I-JEPA [2301.08243](https://arxiv.org/abs/2301.08243), V-JEPA [2404.08471](https://arxiv.org/abs/2404.08471), V-JEPA 2 [2506.09985](https://arxiv.org/abs/2506.09985), Genie [2402.15391](https://arxiv.org/abs/2402.15391), IRIS [2209.00588](https://arxiv.org/abs/2209.00588), DIAMOND [2405.12399](https://arxiv.org/abs/2405.12399), Coconut [2412.06769](https://arxiv.org/abs/2412.06769), Recurrent depth [2502.05171](https://arxiv.org/abs/2502.05171), Pause tokens [2310.02226](https://arxiv.org/abs/2310.02226), Robot Learning tutorial [2510.12403](https://arxiv.org/abs/2510.12403).

URL shorthand used below: `LR/` = `https://github.com/huggingface/lerobot/blob/75cd2ba0ce9ff47686bb2bd7b7f65840606c02ee/`.

---

## 1. Relation to the existing curriculum

- **Path conventions.** Paths in `web/src/lib/paths.json` carry `id`, `titleEn`, `descriptionEn`, `group` (`foundations` / `architectures` / `training` / `systems` / `agents`), `problems` and `prerequisites`. This path fits `architectures` (next to `vision-transformer`, which already has `flow_matching`, `ddim_step`, `adaln_zero`).
- **`embodied_action_stream`** (streaming-llm path, `torch_judge/tasks/embodied_action_stream.py`): a SIMA 2 text-action **parser + discrete-event scheduler** with chunk preemption and held-key release. It cites action chunking (ACT, pi0) only as a model connection. The new VLA module stays on the **numerical** side (binning, DCT, ensembling, flow matching, guidance weights) and must **not** add another chunk-preemption scheduler. LeRobot's async inference queue ([blog](https://huggingface.co/blog/async-robot-inference), [SmolVLA blog "Asynchronous Inference"](https://huggingface.co/blog/smolvla)) is deliberately left out for that reason.
- **`posed_frame_context`** (`torch_judge/tasks/posed_frame_context.py`): RTFM spatial-memory **retrieval** for a frame-generating world model. The new World Models module covers the **learning objectives and planning** of world models (RSSM losses, JEPA latent prediction, ST attention, CEM planning), not retrieval, so there is no overlap.
- **`flow_matching`** (vision-transformer path): uses the noise-at-zero convention (`target = x1 - x0`, `t` unused in the loss). pi0/openpi use the **opposite** convention (noise at `t=1`, integrate `1 -> 0`), which LeRobot now names explicitly (`FlowConvention`, `LR/src/lerobot/policies/common/flow_matching.py` line 49). The new exercise builds on the old one and teaches the convention flip; it does not repeat the loss.
- **Vendoring.** Oracles follow `torch_judge/harness/rl/vendored/`: module docstring with Upstream / Commit / Files / Symbols / Retrieved / License, a "what was changed" section, and no import of `torch_judge.tasks`. A new sibling package (e.g. `torch_judge/harness/embodied/vendored/`) would mirror this.
- **Scope memory.** New exercises are English only and need the author-quality gate (reference passes, mutations rejected in `tests/quality/test_*_mutations.py`, vendored-oracle comparison) per `CONTEXT.md`.

---

## 2. What the public tutorials teach, and in what order

### 2.1 Module 1: VLA

| # | Source | What it teaches, in order | URL |
|---|---|---|---|
| T1 | HF "Robot Learning: A Tutorial" (Capuano et al., interactive Space + arXiv) | Sections in order: LeRobotDataset → Classical Robotics → Robot (RL) Learning (HIL-SERL code example) → Robot (Imitation) Learning: generative models (VAE, Diffusion, **Flow Matching**) → **ACT** → **Diffusion Policy** → Optimized/**async inference** → Generalist Robot Policies: VLMs for VLAs → **pi0** → **SmolVLA** | https://huggingface.co/spaces/lerobot/robot-learning-tutorial (source `app/src/content/article.mdx`, headings read 2026-10-12), https://arxiv.org/abs/2510.12403 |
| T2 | HF Robotics Course | Unit 0 welcome, Unit 1 LeRobot, Unit 2 classical robotics are released; Units 5 RL, 6 Imitation Learning, 7 Foundation Models are "Coming soon". Course is a condensed version of T1. | https://huggingface.co/learn/robotics-course/en/unit0/1 |
| T3 | LeRobot docs, "Policies" section | Order in `docs/source/_toctree.yml`: ACT, SmolVLA, pi0, pi0-FAST, pi0.5, then many newer VLAs (MolmoAct2, VLA-JEPA, GR00T, X-VLA, ...). "Inference" section: deployment, async inference, **Real-Time Chunking (RTC)**. "Robot Processors": **Action Representations** (absolute / relative / delta). | https://huggingface.co/docs/lerobot/act, https://huggingface.co/docs/lerobot/pi0fast, https://huggingface.co/docs/lerobot/rtc, https://huggingface.co/docs/lerobot/action_representations |
| T4 | HF blog "pi0 and pi0-FAST" | Flow matching at 50 Hz; the **2D prefix/suffix block attention mask** and how to build a FlexAttention block mask; **action tokenization**: per-dimension binning problems → FAST (normalize q01/q99 to [-1,1], per-dim DCT, scale-and-round, flatten low frequencies first across dims, BPE); pi0-FAST "5x faster training". | https://huggingface.co/blog/pi0 |
| T5 | HF blog "SmolVLA" | VLM + ~100M flow-matching action expert; 64 visual tokens per frame via PixelShuffle; **layer skipping** (action expert attends to VLM features up to half the layers); **interleaved cross-/self-attention**; async inference. | https://huggingface.co/blog/smolvla |
| T6 | Model cards | OpenVLA-7B (MIT) https://huggingface.co/openvla/openvla-7b ; pi0 (apache-2.0) https://huggingface.co/lerobot/pi0 ; SmolVLA base (apache-2.0) https://huggingface.co/lerobot/smolvla_base ; GR00T N1 2B (NVIDIA non-commercial weights license; "flow matching transformer ... interleaves self-attention over proprioception and actions with cross-attention to the vision and language embeddings", DiT with AdaLN) https://huggingface.co/nvidia/GR00T-N1-2B | — |
| T7 | Papers | RT-2: continuous dims "discretized into 256 bins uniformly", 256 reserved/least-frequent tokens (https://arxiv.org/html/2307.15818); FAST: q01/q99 normalization, DCT, BPE, "interleaving action dimensions by including all low-frequency components first" (https://arxiv.org/html/2501.09747) | — |

GR00T blog: `https://huggingface.co/blog/gr00t-n1` returns 404. Two NVIDIA-namespace HF posts exist (https://huggingface.co/blog/nvidia/gr00t-n1-5-so101-tuning, https://huggingface.co/blog/nvidia/nvidia-isaac-gr00t-in-lerobot), but their content was **not read** (see Section 5).

**Teaching order the tutorials converge on:** action representation and tokenization → action chunking (ACT) → generative action heads (diffusion → flow matching) → VLM backbone + action expert (pi0, SmolVLA, GR00T) → real-time execution (async, RTC).

### 2.2 Module 2: World Models

| # | Source | What it teaches | URL |
|---|---|---|---|
| W1 | DreamerV3 paper | RSSM (sequence model, encoder, dynamics predictor); prediction loss with **symlog** squared loss and **symexp twohot** loss on exponentially spaced bins; **KL balancing** (`L_dyn` with stop-grad on posterior, `L_rep` with stop-grad on prior), **free bits** clipping below 1 nat; categoricals as **1% uniform + 99% network** mixtures; return normalization by the **5th–95th percentile range**, EMA-smoothed. Loss weights: v2 HTML says β_pred=1, **β_dyn=1**, β_rep=0.1; v1 PDF says **β_dyn=0.5**. | https://arxiv.org/html/2301.04104, https://arxiv.org/pdf/2301.04104v1 |
| W2 | DreamerV3 code (JAX) and PyTorch port | The same pieces as code (Section 3.2). Port defaults: `dyn_scale: 0.5`, `rep_scale: 0.1`, `kl_free: 1.0`, `unimix_ratio: 0.01` (`configs.yaml` lines 57–61); upstream default: `loss_scales: {..., dyn: 1.0, rep: 0.1, ...}` (`dreamerv3/configs.yaml` line 86). | https://github.com/danijar/dreamerv3, https://github.com/NM512/dreamerv3-torch |
| W3 | I-JEPA / V-JEPA / V-JEPA 2 | Predict **target-encoder features** of masked blocks from a context encoder; **EMA target encoder** with linear momentum schedule; multi-block masks (2D for I-JEPA, 3D spatio-temporal "tubes" for V-JEPA). I-JEPA code uses `F.smooth_l1_loss` on layer-normed targets; V-JEPA 2 uses `mean(|z-h|^loss_exp)/loss_exp` with `loss_exp: 1.0`. V-JEPA 2-AC adds an action-conditioned predictor (teacher-forced + autoregressive rollout losses) and **CEM planning** to a goal embedding. | https://arxiv.org/abs/2301.08243, https://arxiv.org/abs/2404.08471, https://arxiv.org/abs/2506.09985 |
| W4 | Genie paper | Spatiotemporal (ST) transformer: spatial attention within a frame, **temporal attention with a causal mask** across time; latent action model with VQ codebook **\|A\| = 8**; dynamics model is **MaskGIT**, trained with Bernoulli masking at a rate drawn uniformly from 0.5 to 1. No official code. | https://arxiv.org/html/2402.15391 |
| W5 | DIAMOND / IRIS | DIAMOND: diffusion world model with EDM preconditioning (`c_in`, `c_out`, `c_skip`, `c_noise`). IRIS: discrete autoencoder + autoregressive transformer world model (GPL-3.0, reference only). | https://arxiv.org/abs/2405.12399, https://arxiv.org/abs/2209.00588 |
| W6 | Hugging Face | Transformers ships a V-JEPA 2 model page (https://huggingface.co/docs/transformers/model_doc/vjepa2, exists; contents not reviewed). LeRobot ships **TD-MPC** (latent world model + CEM/MPPI planning, `LR/src/lerobot/policies/tdmpc/modeling_tdmpc.py`) and **VLA-JEPA** (Qwen3-VL + V-JEPA2 world-model loss used only at training time + flow-matching DiT action head; https://huggingface.co/docs/lerobot/vla_jepa). No HF course unit on world models was found. | — |

**Teaching order:** reconstruction-based latent dynamics (DreamerV3 RSSM and its loss tricks) → non-generative latent prediction (JEPA) → using a world model to act (planning with CEM / MPPI) → generative interactive world models (Genie, DIAMOND).

### 2.3 Module 3: Latent Reasoning

| # | Source | What it teaches | URL |
|---|---|---|---|
| L1 | Pause tokens (Goyal et al.) | Append learnable `<pause>` tokens to the input prefix; ignore outputs until the last pause; during training ignore the loss on positions whose **next** token is `<pause>` (`S_ignore = {k : p_{k+1} = <pause>}`). No official code found. | https://arxiv.org/html/2310.02226 |
| L2 | Coconut | Feed the **last hidden state** back as the next input embedding ("continuous thought"), between `<bot>` and `<eot>`; **multi-stage curriculum**: at stage k replace the first k language steps with k·c thoughts; GSM8k uses **c = 2** and 3 stages plus a final stage. | https://arxiv.org/html/2412.06769 |
| L3 | Recurrent depth (Huginn) | Prelude → core block iterated r times → coda; adapter `A: R^{2h} -> R^h` on the **concatenation** of state and prelude embedding; r drawn from a **log-normal Poisson** (σ = 1/2); truncated backprop through the last k iterations. | https://arxiv.org/html/2502.05171 |
| L4 | Huginn model card | `num_steps` controls test-time compute; per-token adaptive exit criteria `entropy-diff`, `latent-diff`, `kl`, `argmax-stability`, `exit_threshold`; needs its own KV cache. | https://huggingface.co/tomg-group-umd/huginn-0125 |
| L5 | Hugging Face tutorials | **None found.** A web search for an HF blog on Coconut / recurrent depth / pause tokens returned only papers and third-party commentary. | — |

**Teaching order:** extra compute via inert tokens (pause) → recurrence through the sequence with continuous thoughts (Coconut) → recurrence in depth with adaptive exit (Huginn).

---

## 3. Candidate exercises

All candidates are pure functions or tiny modules on small tensors, CPU-only, seeded, no pretrained weights, no simulators. "Oracle" means the upstream symbol to transcribe into a vendored module. Difficulty uses the existing labels (Easy / Medium / Hard).

### 3.1 Module 1: VLA

**V1. `vla_action_bins` — uniform action binning, two conventions** (Medium)
- Contract: `bin_actions(actions, low, high, n_bins, convention) -> (tokens, recon)`; `actions (..., D)` float; `low/high (D,)` from q01/q99. Normalize `2*(x-low)/(high-low+1e-8)-1` clipped to [-1,1]; dims with `low == high` map to 0. `convention="openvla"`: `np.digitize` over `linspace(-1,1,n_bins)` edges, decode to the `n_bins-1` **bin centers** with index clip; `"rt1"`: `int((a-min)/(max-min)*(V-1))`, decode `a/(V-1)`. Token ids map to the vocab tail (`vocab_size - bin`).
- Edge cases: value exactly at +1 (digitize returns `n_bins`, decodes to last center), values outside [q01,q99], constant dims, decode of an out-of-range id.
- Oracle: OpenVLA `prismatic/vla/action_tokenizer.py` `ActionTokenizer.__call__`, `decode_token_ids_to_actions` (https://github.com/openvla/openvla/blob/c8f03f48af692657d3060c19588038c7220e9af9/prismatic/vla/action_tokenizer.py); normalization `prismatic/vla/datasets/rlds/utils/data_utils.py::normalize_action_and_proprio` (line 61, TF); unnormalize in `prismatic/extern/hf/modeling_prismatic.py` (around line 528). RT-1 `tokenizers/action_tokenizer.py::tokenize/detokenize` (lines 103, 129; TF, Apache-2.0).
- Mutations: decode to bin edges instead of centers; forget the `-1`/clip so the top bin indexes out of range; rescale by min/max instead of q01/q99 or skip the constant-dim zeroing.

**V2. `fast_dct_action_tokens` — FAST before BPE** (Medium)
- Contract: `fast_encode(chunk, scale=10.0, min_token=0) -> LongTensor` and `fast_decode(ints, T, D, scale) -> (T, D)`; `chunk (T, D)` or `(B, T, D)`. Orthonormal DCT-II along time, `round(coeff*scale)`, flatten **row-major** of the `(T, D)` coefficient matrix (low frequency first, dims interleaved), subtract `min_token`, clamp at 0. Decode: reshape, add `min_token`, divide by scale, orthonormal inverse DCT. Implement DCT as a matrix in torch.
- Edge cases: T = 1; constant chunk (only the DC row non-zero); negative coefficients below `min_token` (clamped, lossy); round-half behavior (`np.around` is half-to-even, as is `torch.round`).
- Oracle: `processing_action_tokenizer.py` `__call__` (lines 43–58), `decode` (60–97), `fit` (99+) at https://huggingface.co/physical-intelligence/fast/blob/ec4d7aa71691cac0b8bed6942be45684db2110f4/processing_action_tokenizer.py. Uses `scipy.fft.dct/idct`; BPE excluded (existing `bpe` exercise covers it). openpi wraps it in `src/openpi/models/tokenizer.py::FASTTokenizer` (line 51), mapping tokens to the PaliGemma vocab tail and discretizing state into 256 bins.
- Mutations: DCT over the action-dim axis instead of time; column-major flatten (all frequencies of dim 0 first); non-orthonormal DCT (wrong scale on round-trip); forgetting `/scale` on decode.

**V3. `act_temporal_ensemble` — online temporal ensembling** (Medium)
- Contract: class `TemporalEnsembler(coeff, chunk_size)` with `reset()` and `update(actions (B, chunk, D)) -> (B, D)`. Weights `w_i = exp(-coeff * i)`, `w_0` = oldest prediction; each call merges the new chunk into a running weighted average for every future step, appends the last step, and pops step 0.
- Edge cases: first call (no history); `coeff = 0` (uniform), negative coeff (newer weighted more); counts saturating at `chunk_size`; batch > 1; must equal the offline weighted average over all chunks that cover the step.
- Oracle: `LR/src/lerobot/policies/act/modeling_act.py::ACTTemporalEnsembler` (line 170; default coeff 0.01 per docstring). Offline original: `imitate_episodes.py` lines 250–259 at https://github.com/tonyzhaozh/act/blob/742c753c0d4a5d87076c8f69e5628c79a8cc5488/imitate_episodes.py (detects "populated" rows with `!= 0`, so a true zero action is dropped — an upstream quirk to avoid copying).
- Mutations: weight index reversed (newest gets `w_0`); dividing by the count instead of cumulative weights; not popping the consumed step (off-by-one in time).

**V4. `pi0_flow_matching_chunk` — pi0 training inputs and Euler sampling** (Medium)
- Contract: `make_inputs(actions (B,H,D), noise, time (B,), convention, prefix_mask=None) -> (x_t, v_target, model_time)` and `euler_sample(velocity_fn, noise, num_steps, convention, hard_prefix=None, hard_prefix_mask=None) -> (B,H,D)`. `noise_at_one`: `x_t = t*noise + (1-t)*actions`, `v = noise - actions`, `dt = -1/N`, `t_k = 1 + k*dt`. `noise_at_zero`: the mirror. Prefix positions get the clean-end time and are clamped before and after each step.
- Edge cases: `num_steps = 1`; shape validation (`time` must be `(B,)`); prefix mask all-true / all-false; with an exact linear velocity field the sampler must recover `actions` exactly.
- Oracle: `LR/src/lerobot/policies/common/flow_matching.py::make_flow_matching_inputs` (line 188), `euler_integrate` (line 256), `sample_time_beta` (line 167; pi-family recipe `Beta(1.5, 1)*0.999 + 0.001`, `LR/src/lerobot/policies/pi0/configuration_pi0.py` lines 47–50). JAX original: `src/openpi/models/pi0.py::Pi0.compute_loss` (`time = beta(1.5,1)*0.999+0.001`, `x_t = t*noise+(1-t)*actions`, `u_t = noise - actions`) and `sample_actions` (`dt = -1/num_steps`), https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/models/pi0.py.
- Mutations: mixing conventions (noise-at-one interpolation with noise-at-zero target); `dt` with the wrong sign; prefix clamped only after the step, not before.

**V5. `rtc_prefix_guidance` — Real-Time Chunking weights and guidance clamp** (Hard, capstone)
- Contract: `prefix_weights(start, end, total, schedule) -> (total,)` for `zeros | ones | linear | exp`, and `guided_velocity(x_t, v_fn, prev_chunk, time, inference_delay, execution_horizon, max_guidance_weight, schedule) -> v`. Weights: 1 for the first `inference_delay` steps (already committed), a decaying ramp up to `execution_horizon`, 0 after. Guidance: predict `x1 = x_t - time*v`, error `(prev - x1)*w`, vector-Jacobian product via `torch.autograd.grad`, scale by `min(c*inv_r2, max_w)` with `tau = 1 - time`.
- Edge cases: `start >= end`; `end > total`; previous chunk shorter than the new chunk (zero-padded); `time` at the endpoints (division by zero handled by `nan_to_num` + clamp); `prev_chunk is None` returns the plain velocity.
- Oracle: `LR/src/lerobot/policies/rtc/modeling_rtc.py::RTCProcessor.get_prefix_weights` (line 256) and `denoise_step` (line 122), which cites https://github.com/Physical-Intelligence/real-time-chunking-kinetix/blob/main/src/model.py#L214 (MIT; commit not pinned). Paper: https://arxiv.org/abs/2506.07339.
- Mutations: `linspace(1,0,n)` without dropping endpoints; forgetting `1 - time` (wrong time convention); no clamp at `max_guidance_weight`.

Other VLA candidates (not in the recommended list):

| Id | What | Oracle | Difficulty | Why not recommended |
|---|---|---|---|---|
| `vla_prefix_block_mask` | `make_att_2d_masks(pad, ar)`: `cumsum(ar)[:,None,:] <= cumsum(ar)[:,:,None]` AND pad outer product; positions `cumsum(pad)-1`; inference suffix-to-prefix mask | `LR/src/lerobot/policies/common/vla_utils.py` line 60 ("exact copy" of openpi/big_vision); openpi `pi0.py::make_attn_mask` line 19 | Easy–Medium | Good warm-up; could be part 1 of V4 instead of its own exercise |
| `spatial_softmax_keypoints` | softmax over H·W per channel, expected (x,y) on a numpy-built `[-1,1]` grid | `LR/src/lerobot/policies/diffusion/modeling_diffusion.py::SpatialSoftmax` line 419 | Easy | Vision, not VLA-specific |
| `film_residual_block_1d` | Diffusion Policy conditional 1D residual block with FiLM bias (and optional scale) | same file, line 795 | Easy | Close to existing `adaln_zero` |
| `vqbet_residual_vq` | residual VQ with straight-through and commitment loss | `LR/src/lerobot/policies/vqbet/vqbet_utils.py::ResidualVQ` line 323 (ported from lucidrains/vector-quantize-pytorch per comment line 41) | Medium | VQ is reused better in Genie's latent actions; overlaps |
| `relative_action_chunk` | absolute ↔ relative (all from current state) ↔ delta (cumulative) | https://huggingface.co/docs/lerobot/action_representations | Easy | Too small alone; good visible example inside V1 |

### 3.2 Module 2: World Models

**W1. `symlog_twohot` — DreamerV3 value/reward head** (Medium)
- Contract: `symlog(x)`, `symexp(x)`; `twohot_target(y, bins) -> (..., K)` distributing mass between the two nearest bins in symlog space; `twohot_loss(logits, y, bins)`; `twohot_mean(logits, bins)` computed as a **symmetric** sum so that uniform probabilities over symmetric bins give exactly 0. Bins: 255 symmetric values in [-20, 20].
- Edge cases: target exactly on a bin (`below == above`, weight 1); target outside the bin range (clipped to edge); y = 0; even vs odd K.
- Oracle: `embodied/jax/nets.py::symlog/symexp` (lines 59–64), `embodied/jax/outs.py::TwoHot` (line 273), bins built in `embodied/jax/heads.py` (lines 132–144), https://github.com/danijar/dreamerv3/tree/e01491fad6434b2245a3b8ca201dd7faedcc458c. PyTorch: `tools.py::symlog` (line 23), `DiscDist` (line 452) at https://github.com/NM512/dreamerv3-torch/blob/6ef8646d807cd10ce0c88e10a7e943211e7fc44c/tools.py (bins from `linspace(-20,20,255)`; mean uses a plain sum).
- Mutations: two-hot in raw space instead of symlog space; swapping `weight_below`/`weight_above`; plain left-to-right sum (non-zero at init, caught by an exact-zero check).

**W2. `rssm_kl_balance` — categorical latents, KL balancing, free bits** (Medium)
- Contract: `rssm_kl_loss(post_logits, prior_logits, free_nats=1.0, beta_dyn=1.0, beta_rep=0.1, unimix=0.01) -> (loss, dyn, rep)`; logits `(B, T, S, C)` (S groups of C classes). Apply unimix (`(1-u)*softmax + u/C`, back to log), KL per group summed over S, then `max(free_nats, ·)` per (B,T), stop-gradient on the posterior for `dyn` and on the prior for `rep`. Optional part 2: straight-through one-hot sample `onehot + probs - sg(probs)`.
- Edge cases: identical post/prior (KL = 0, clipped to free_nats, **zero gradient**); `unimix = 0`; gradient routing (dyn's gradient reaches only prior logits, rep's only posterior logits) checked with autograd.
- Oracle: `dreamerv3/rssm.py::RSSM.loss` (lines 120–131) and `_dist` (Agg sum over groups, line 173), `outs.py::Categorical` (unimix, KL; line 208); defaults `dreamerv3/configs.yaml` lines 86 and 91. PyTorch: `networks.py::RSSM.kl_loss` (line 272) and `tools.py::OneHotDist` (line 425).
- Mutations: free bits applied per group before summing; stop-gradient on the wrong side; unimix applied to logits instead of probabilities.

**W3. `jepa_block_masks_and_loss` — multi-block masking, latent loss, EMA target** (Medium)
- Contract: `make_masks(grid=(T,H,W), blocks=[(t0,t,top,h,left,w), ...] per sample) -> (ctx_idx, tgt_idx)` where target = union of blocks and context = complement, each list **truncated to the batch minimum length**; `jepa_loss(pred, target_feats, tgt_idx, loss_exp=1.0)` with `layer_norm` on targets and `mean(|z-h|^p)/p`; `ema_update(target_params, online_params, m)` in place.
- Edge cases: overlapping blocks; block that leaves an empty context (upstream resamples; the exercise should raise); `max_keep` cap; momentum schedule endpoints.
- Oracle: `src/masks/multiseq_multiblock3d.py::_MaskGenerator` (`_sample_block_size` line 145, `_sample_block_mask`, `__call__` truncation to `min_keep`), `src/masks/utils.py::apply_masks`, `app/vjepa/train.py` (momentum schedule lines 296–303, `loss_fn` lines 440–449, EMA lines 472–480), https://github.com/facebookresearch/vjepa2/tree/204698b45b3712590f06245fbfba32d3be539812 (MIT). Block placement is passed in, so the exercise has no RNG.
- Mutations: no layer norm on targets; EMA as `m*online + (1-m)*target` (swapped); not truncating to the batch minimum, so the batch cannot be stacked.

**W4. `st_transformer_attention` — Genie spatiotemporal attention** (Hard)
- Contract: `st_attention(x (B,T,N,D), Wq, Wk, Wv, Wo, n_heads) -> (B,T,N,D)`: spatial attention within each frame (bidirectional over N), then temporal attention per patch position over T with a **causal** mask; plus `st_mask(T, N)` giving the equivalent boolean masks on the flattened `T*N` sequence. Test: factorized result equals masked full attention with those masks.
- Edge cases: T = 1 (temporal is identity on values); N = 1; future frame perturbation must not change earlier outputs.
- Oracle: open-genie `genie/module/attention.py::SpaceTimeAttention` (line 373; spatial `causal=False`, temporal `causal=True`), https://github.com/myscience/open-genie/blob/732b9f9b746f18fff1a0fb22f83638224f2f7cc6/genie/module/attention.py (MIT, **unofficial**). Spec: Genie paper ST-transformer paragraph (https://arxiv.org/html/2402.15391).
- Mutations: temporal attention not causal; spatial attention causal; attention over the flattened T·N sequence (no factorization).

**W5. `world_model_cem_step` — planning refit with a world model** (Medium, capstone)
- Contract: `cem_refit(samples (S,H,A), energies (S,), mean, std, topk, momentum_mean, momentum_std, mode) -> (mean, std)`. `mode="vjepa2"`: top-k **lowest** energy (L1 distance to goal embedding), unweighted mean and **unbiased** std, momentum blend `new*(1-m) + old*m`. `mode="tdmpc"`: top-k **highest** value, softmax weights `exp(temp*(v - max v))`, weighted mean/std. A part 2 can roll a provided toy linear world model to compute energies.
- Edge cases: ties in energy (stable order); `topk = 1` (std of one sample: NaN with unbiased std — spec must decide); `topk = S`.
- Oracle: `notebooks/utils/mpc_utils.py::cem` (line 28) in vjepa2 (MIT); `LR/src/lerobot/policies/tdmpc/modeling_tdmpc.py::TDMPCPolicy.plan` (line 167; Apache-2.0, header credits Nicklas Hansen et al.).
- Mutations: `topk(largest=True)` on an energy; momentum applied the wrong way round; softmax without subtracting the max (overflow on large values).

Other World Models candidates:

| Id | What | Oracle | Difficulty | Note |
|---|---|---|---|---|
| `dreamer_return_norm` | lambda-return + percentile(5/95) EMA scale with `max(1, hi-lo)` | danijar `dreamerv3/agent.py::lambda_return` (line 482), `embodied/jax/utils.py::Normalize` (line 16); NM512 `models.py::RewardEMA` (line 11) | Medium | Overlaps `gae_advantage`, `reinforce_discounted_returns`; good alternate for W5 |
| `maskgit_schedule_step` | MaskGIT step: tokens-per-step schedule (linear/cosine/arccos), fill top-confidence masked tokens | open-genie `genie/dynamics.py::get_schedule` (line 167) and `generate` | Medium | Unofficial reference; the upstream linear schedule starts at 0 and relies on `clamp(min=1)` |
| `vjepa2_action_block_mask` | frame-block causal mask for interleaved action/state/patch tokens | vjepa2 `src/models/utils/modules.py::build_action_block_causal_attention_mask` (line 12) | Easy | Same idea as `vla_prefix_block_mask` |
| `edm_preconditioning` | `c_in, c_out, c_skip, c_noise` and the training target | DIAMOND `src/models/diffusion/denoiser.py::compute_conditioners` (line 66), MIT | Easy | Fits better in vision-transformer path's diffusion section |

### 3.3 Module 3: Latent Reasoning

**R1. `pause_token_training_mask`** (Easy)
- Contract: `insert_pauses(prefix_ids, target_ids, pause_id, n_pause) -> (input_ids, labels)`; labels `-100` on prefix positions and on every position whose next token is `<pause>`; return also `answer_start` (first index after the last pause).
- Edge cases: `n_pause = 0` (plain SFT mask); empty target; `pause_id` appearing in the prefix.
- Oracle: **none**. Only the paper's `S_ignore` definition and loss (https://arxiv.org/html/2310.02226, "pause-training loss"). No official code was found (Section 5).
- Mutations: masking the pause positions themselves instead of the positions predicting them (off-by-one); computing loss on pause targets; prepending instead of appending.

**R2. `coconut_stage_sample` — curriculum sample builder** (Medium)
- Contract: `build_stage_sample(question, steps, answer, stage, c_thought, max_latent_stage, pad_latent_to_max, ids) -> dict(input_ids, labels, position_ids)`. Stage k (≤ max) skips the first k steps and inserts `k*c` latent ids between `<bot>`/`<eot>`; stage > max skips **all** steps and uses `max` (or `min(len(steps), max)`) latent groups; labels `-100` over question + markers + latents.
- Edge cases: stage 0 (plain CoT with empty latent span); stage > number of steps; `no_special_marker`; `pad_latent_to_max` toggling.
- Oracle: `dataset.py::get_cot_latent_dataset.process_dataset` (line 230), https://github.com/facebookresearch/coconut/blob/27273cb8cca4bb763c041a63b036d0c3b7cbbb48/dataset.py (MIT), with `uniform_prob = 0` for determinism; GSM settings in `args/gsm_coconut.yaml` (`c_thought: 2`, `max_latent_stage: 3`, `pad_latent_to_max: True`).
- Mutations: `n_latent = k` instead of `k*c`; not skipping all steps once `stage > max_latent_stage`; label offset that forgets the two marker tokens.

**R3. `coconut_latent_forward` — continuous-thought feedback** (Hard)
- Contract: `coconut_forward(model, embed, input_ids, latent_id) -> logits` where `model(inputs_embeds) -> (logits, last_hidden)` is a provided tiny causal model (e.g. one `gpt2_block`). For pass i, the embedding at the i-th latent position is replaced by the last hidden state at position-1; passes = `max_n_latents + 1`; batch items may have different latent counts. A naive full-recompute reference and the KV-cached version must agree.
- Edge cases: no latent tokens (one pass); latents at different offsets per batch item; gradient flows through the fed-back hidden states (check with autograd).
- Oracle: `coconut.py::Coconut.forward` (lines 40–190), same repo/commit (MIT). Upstream depends on the HF GPT-2 interface (`past_key_values`, `output_hidden_states`), so vendoring needs a stub model with that interface.
- Mutations: feed the hidden state at the latent's own position (not position-1); fill all latents in one pass; detach the fed-back state.

**R4. `recurrent_depth_unroll` — prelude/core/coda with truncated backprop** (Medium–Hard)
- Contract: `recurrent_forward(e, s0, adapter, core, n_no_grad, k_grad) -> s` with `s <- core(adapter(cat[s, e]))`; first `n_no_grad` iterations under `no_grad`, last `k_grad` with gradient; and `split_iterations(p, mean_backprop_depth) -> (n, k) = (max(p - s, 0), min(s, p))` given a sampled total `p` (the log-normal Poisson draw is passed in, so no RNG).
- Edge cases: `k_grad = 0` (eval: no core gradient); `p < s`; gradient check: core parameters receive exactly the gradient of the last k unrolled steps; the prelude still gets gradient through `e` every step.
- Oracle: `recpre/raven_modeling_minimal.py::iterate_forward` (line 727), `core_block_forward` (line 764), `randomized_iteration_sampler` (line 784), `initialize_state` (line 804), https://github.com/seal-rg/recurrent-pretraining/blob/1ea7220ec7eb42d13e89db0663df254d0bcdc28e/recpre/raven_modeling_minimal.py (Apache-2.0). Defaults in `recpre/raven_config_minimal.py`: `mean_recurrence 32`, `mean_backprop_depth 8`.
- Mutations: addition instead of concatenation into the adapter (shape mismatch caught, but also an "add then project" variant); `no_grad` on the last k instead of the first n; injecting `e` only at step 0.

**R5. `latent_exit_criteria` — per-token adaptive compute** (Medium)
- Contract: `first_exit_step(logits_per_step (R, N, V), latents_per_step (R, N, h), criterion, threshold) -> LongTensor (N,)` for `kl` (KL of **previous** to current: `F.kl_div(log_p, prev_log_p, log_target=True).sum(-1)`), `entropy-diff`, `latent-diff` (`||s - s_prev|| / ||s||`), `argmax-stability`; tokens that never exit return R.
- Edge cases: exit at step 1 vs initial sentinel (entropy-diff starts from 100.0); threshold "auto" defaults (`1e-3` entropy, `0.03` latent, `0.001` kl); ties.
- Oracle: `raven_modeling_minimal.py` exit evaluators (`EntropyDiffExitEvaluator` line 2128, `LatentDiffExitEvaluator` 2149, `KLExitEvaluator` 2167, `ArgmaxStabilityExitEvaluator` 2218), same commit. Card: https://huggingface.co/tomg-group-umd/huginn-0125.
- Mutations: KL direction reversed; absolute instead of relative latent change; returning the first step **after** the condition instead of the step where it holds.

---

## 4. Recommended ordered list

Path proposal: id `vla-world-models-latent-reasoning`, group `architectures`, path-level `prerequisites: ["transformer-internals"]` (advisory).

**Entry baseline** (per `CONTEXT.md`): tensor ops, small `nn.Module`s, autograd, ordinary multi-head attention. Not assumed: robotics, diffusion, RL.

| Order | Module | Exercise | Difficulty | Advisory prerequisites (existing ids) |
|---|---|---|---|---|
| 1 | VLA | `vla_action_bins` | Medium | `cross_entropy`, `embedding` |
| 2 | VLA | `fast_dct_action_tokens` | Medium | `vla_action_bins`, `bpe` |
| 3 | VLA | `act_temporal_ensemble` | Medium | `embodied_action_stream` (chunks, contrast) |
| 4 | VLA | `pi0_flow_matching_chunk` | Medium | `flow_matching`, `ddim_step`, `causal_attention` |
| 5 | VLA | `rtc_prefix_guidance` | Hard | `pi0_flow_matching_chunk`, `act_temporal_ensemble` |
| 6 | World Models | `symlog_twohot` | Medium | `cross_entropy`, `label_smoothing` |
| 7 | World Models | `rssm_kl_balance` | Medium | `k3_kl_penalty`, `softmax_entropy` |
| 8 | World Models | `jepa_block_masks_and_loss` | Medium | `vit_patch`, `layernorm`, `contrastive_loss` |
| 9 | World Models | `st_transformer_attention` | Hard | `mha`, `causal_attention`, `posed_frame_context` (contrast) |
| 10 | World Models | `world_model_cem_step` | Medium | `jepa_block_masks_and_loss`, `gae_advantage` |
| 11 | Latent Reasoning | `pause_token_training_mask` | Easy | `response_token_mask` |
| 12 | Latent Reasoning | `coconut_stage_sample` | Medium | `pause_token_training_mask` |
| 13 | Latent Reasoning | `coconut_latent_forward` | Hard | `gpt2_block`, `kv_cache` |
| 14 | Latent Reasoning | `recurrent_depth_unroll` | Medium–Hard | `mamba_ssm` (recurrence), `gradient_accumulation` |
| 15 | Latent Reasoning | `latent_exit_criteria` | Medium | `masked_kd_kl`, `softmax_entropy` |

Per module, names only: VLA = bins, FAST, ACT ensemble, pi0 flow chunk, RTC; World Models = symlog/two-hot, RSSM KL, JEPA masks, ST attention, CEM step; Latent Reasoning = pause mask, Coconut stage, Coconut forward, recurrent unroll, exit criteria.

---

## 5. Risks and open questions

### Licenses that block vendoring
- **I-JEPA and V-JEPA v1 are CC BY-NC 4.0** (`LICENSE` in facebookresearch/ijepa and facebookresearch/jepa). Do not vendor; use V-JEPA 2 (MIT) as the oracle. The I-JEPA `smooth_l1_loss` variant can be described with a link, not copied.
- **IRIS is GPL-3.0.** Reference only.
- **GR00T N1 weights** carry the NVIDIA non-commercial license (model card); Isaac-GR00T code is Apache-2.0 per the GitHub API. No exercise needs the weights.
- **openpi ships `LICENSE_GEMMA.txt`.** Vendor only the helper arithmetic (masks, flow-matching inputs), not Gemma-derived model files; confirm which files that license covers before vendoring anything from `models/gemma*.py`.
- **V-JEPA 2**: three named files are Apache-2.0, the rest MIT (README "License"); none of the three are used here.

### Oracle gaps and transcription risks
- **No upstream code for pause tokens** (R1) or RT-2. R1 can only be checked against the paper formula; it fails the "vendored oracle" layer of the author-quality gate unless the gate allows a paper-only oracle.
- **Genie has no official code.** open-genie is an unofficial MIT reimplementation; using it as the "vendored oracle" trusts a third party. Spec from the paper, and use open-genie only as a cross-check.
- **DreamerV3 upstream is JAX**, so a "line for line" transcription becomes JAX → PyTorch. The MIT PyTorch port is closer but diverges: `dyn_scale 0.5` vs upstream `dyn: 1.0`; plain-sum mean vs symmetric sum; bins from one `linspace` vs two mirrored halves. The paper itself changed β_dyn from 0.5 (v1) to 1 (v2). The exercise should take β values as parameters.
- **FAST oracle needs `scipy.fft`**, which is not a `pyre-code` dependency (`pyproject.toml` lists fastapi, uvicorn, torch, numpy<2, pydantic). Options: precomputed fixtures from scipy, or a DCT-matrix oracle checked once offline.
- **Coconut forward** relies on the HF GPT-2 KV-cache interface; the oracle needs a stub model, which makes it partly author-written.
- **LeRobot moves fast.** `policies/common/flow_matching.py` is a 2026 refactor that consolidated copies from pi0, pi05, smolvla, eo1, groot, evo1 and wall_x (module docstring). Pin `75cd2ba`; links to `main` will drift.

### Ideas too big for one exercise
- A full VLA forward (PaliGemma/SmolVLM + action expert), SmolVLA layer skipping and interleaved CA/SA, GR00T's DiT head: model-sized, need weights to be meaningful. Teach only the masks and the flow-matching head.
- A full RSSM training loop, a Genie tokenizer + LAM + dynamics, a DIAMOND world model: too slow or too many parts. Split into the loss and attention pieces above.
- VQ-BeT end to end (RVQ + GPT + offset head): keep only residual VQ if at all.
- Async inference server/client: overlaps `embodied_action_stream` and is systems, not tensors.

### Things not verified
- Content of the NVIDIA GR00T blog posts on HF (URLs exist, not read); `https://huggingface.co/blog/gr00t-n1` is 404.
- The RTC kinetix repo commit (license from GitHub API only; file `src/model.py#L214` taken from the LeRobot comment).
- Diffusion Policy original repo internals (cloned, license read, code not read); this doc cites LeRobot's port instead.
- Whether a learnable pause-token reference implementation exists anywhere official.
- The HF Robotics Course units 5–7 are "Coming soon", so their order and exercises cannot be checked yet.
- Line numbers are from the pinned clones above; they will not match `main`.

### Top open questions
1. Can an exercise ship with a **paper-only oracle** (pause tokens, Genie ST attention) or must every exercise have a vendored upstream oracle? This decides whether R1 and W4 stay.
2. How should the **FAST DCT oracle** run without scipy: precomputed fixtures, or add scipy as a test-only dependency?
3. Should the path be **one path with three modules** (as proposed) or three paths? Latent reasoning shares little with VLA and world models, and the paths page groups by theme (5 group cards, no duplicate listing).

---

## 6. Decisions (2026-10-12)

1. **Paper-only oracles are allowed** for `pause_token_training_mask` and `st_transformer_attention`. The reference is transcribed from the paper's formulas, and the exercise names the paper as its oracle source. `st_transformer_attention` is also cross-checked against open-genie (unofficial, MIT). All 15 exercises stay.
2. **scipy becomes a test-only dependency**, so the FAST exercise is checked against `scipy.fft.dct` directly. It stays out of the runtime dependencies.
3. **Two paths, not one:**
   - "Embodied AI: VLA and World Models", 10 exercises (VLA 1–5, World Models 6–10), group `architectures`.
   - "Latent Reasoning", 5 exercises (11–15). Its group is decided when the path is added.
