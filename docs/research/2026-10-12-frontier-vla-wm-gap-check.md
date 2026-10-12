# Research: frontier VLA and world-model techniques not yet covered

Date: 2026-10-12
Status: Research input. No exercises were created.
Builds on: `docs/research/2026-10-12-vla-world-models-latent-reasoning.md` (called "the prior doc" below). Its conclusions, pins and decisions still hold. This doc only adds what is missing for **frontier** models (late 2025 to 2026).

How this was researched: upstream repos were shallow-cloned on 2026-10-12 and read directly. The commit is the `HEAD` of each clone; the license comes from that repo's `LICENSE` file. arXiv ids and titles were checked with the arXiv export API (`export.arxiv.org/api/query?id_list=...`). Paper details come from `arxiv.org/html/<id>` (section numbers given). Blog-only claims are marked as such. Nothing here was taken from secondary write-ups.

---

## 0. Sources

### 0.1 Code (new pins, plus LeRobot reused)

| Short name | Repo | Commit (date) | License |
|---|---|---|---|
| LeRobot | https://github.com/huggingface/lerobot | `75cd2ba0ce9ff47686bb2bd7b7f65840606c02ee` (prior doc's pin). Every LeRobot file and line cited below was checked at this pin. `HEAD` on 2026-10-12 is `24f15b92b556cb861ca65e989462e23505a701bd`, with the same line numbers for these files. | Apache-2.0 |
| openpi | https://github.com/Physical-Intelligence/openpi | `215abfb217dbac7d5f1273282331b9b1866c0479` (2026-08-24) | Apache-2.0. `gemma.py` carries a Big Vision Apache-2.0 header. |
| OpenVLA-OFT | https://github.com/moojink/openvla-oft | `e4287e94541f459edc4feabc4e181f537cd569a8` (2025-09-08) | MIT |
| Isaac-GR00T (N1.7, `main`) | https://github.com/NVIDIA/Isaac-GR00T | `d2b7e75b937e3ec9aa5dbc798f08b89692c49734` (2026-10-07) | Apache-2.0 |
| Isaac-GR00T (N1.5, branch `n1d5`) | same | `4af2b622892f7dcb5aae5a3fb70bcb02dc217b96` | Apache-2.0 |
| LAPA | https://github.com/LatentActionPretraining/LAPA | `46aca51d7faebcec02d7d323bbb3820c2df07bc6` (2025-01-22) | MIT |
| UniVLA | https://github.com/OpenDriveLab/UniVLA | `0ab9e9dd3074ffa14fdd1a00aa2aa84e4f8e1352` (2025-11-19) | Apache-2.0 |
| Cosmos-Tokenizer | https://github.com/NVIDIA/Cosmos-Tokenizer | `3584ae752ce8ebdbe06a420bf60d7513c0e878cc` (2025-02-10) | Apache-2.0 |
| vector-quantize-pytorch | https://github.com/lucidrains/vector-quantize-pytorch | `4103bf2c7be993b6e0ac21d88684818c9880c478` (2026-09-23) | MIT |
| Diffusion Forcing | https://github.com/buoyancy99/diffusion-forcing | `475e0bcab87545e48b24b39fb46a81fe59d80594` (2025-02-22) | MIT |
| Self-Forcing | https://github.com/guandeh17/Self-Forcing | `33593df3e81fa3ec10239271dd2c100facac6de1` (2025-09-11) | Apache-2.0 |
| open-oasis | https://github.com/etched-ai/open-oasis | `f59deef2c019c212bd0c5a3a5b986a51f3701847` (2024-11-08) | MIT |
| TD-MPC2 | https://github.com/nicklashansen/tdmpc2 | `e9f59321933cbc8e11a002b842adc7d4ffae8ff1` (2026-07-13) | MIT |
| V-JEPA 2 | https://github.com/facebookresearch/vjepa2 | `204698b45b3712590f06245fbfba32d3be539812` (same as the prior doc) | MIT (with three named Apache-2.0 files) |
| Navigation World Models | https://github.com/facebookresearch/nwm | `3f6cd8e70d6f2d1e2b9684acff510710135f0f41` (2025-08-13) | **CC BY-NC 4.0** |

URL shorthand: `LR/` = `https://github.com/huggingface/lerobot/blob/75cd2ba0ce9ff47686bb2bd7b7f65840606c02ee/`.

There is **no official code** for Dreamer 4. The author's GitHub (`api.github.com/users/danijar/repos`) and the project page https://danijar.com/project/dreamer4/ link no repository. There is also no official code for pi*0.6 / RECAP: openpi's README (line 8) says it "only support[s] the flow matching head" for pi0.5, and a grep of openpi for `advantage`/`recap` finds nothing. Gemini Robotics 1.5, Genie 2/3 and Helix have no code either.

### 0.2 Papers (ids and titles checked with the arXiv API)

pi0.5 [2504.16054](https://arxiv.org/abs/2504.16054) · Knowledge Insulating VLAs [2505.23705](https://arxiv.org/abs/2505.23705) · pi*0.6 [2511.14759](https://arxiv.org/abs/2511.14759) · OpenVLA-OFT ("Fine-Tuning Vision-Language-Action Models: Optimizing Speed and Success") [2502.19645](https://arxiv.org/abs/2502.19645) · GR00T N1 [2503.14734](https://arxiv.org/abs/2503.14734) · FLARE [2505.15659](https://arxiv.org/abs/2505.15659) · SmolVLA [2506.01844](https://arxiv.org/abs/2506.01844) · X-VLA [2510.10274](https://arxiv.org/abs/2510.10274) · Gemini Robotics 1.5 [2510.03342](https://arxiv.org/abs/2510.03342) · Hi Robot [2502.19417](https://arxiv.org/abs/2502.19417) · LAPA [2410.11758](https://arxiv.org/abs/2410.11758) · UniVLA [2505.06111](https://arxiv.org/abs/2505.06111) · FSQ [2309.15505](https://arxiv.org/abs/2309.15505) · Cosmos WFM [2501.03575](https://arxiv.org/abs/2501.03575) · Cosmos-Reason1 [2503.15558](https://arxiv.org/abs/2503.15558) · Cosmos-Predict2.5 ("World Simulation with Video Foundation Models for Physical AI") [2511.00062](https://arxiv.org/abs/2511.00062) · Dreamer 4 ("Training Agents Inside of Scalable World Models") [2509.24527](https://arxiv.org/abs/2509.24527) · Diffusion Forcing [2407.01392](https://arxiv.org/abs/2407.01392) · Self Forcing [2506.08009](https://arxiv.org/abs/2506.08009) · CausVid ("From Slow Bidirectional to Fast Autoregressive Video Diffusion Models") [2412.07772](https://arxiv.org/abs/2412.07772) · GameNGen ("Diffusion Models Are Real-Time Game Engines") [2408.14837](https://arxiv.org/abs/2408.14837) · Navigation World Models [2412.03572](https://arxiv.org/abs/2412.03572) · UniSim ("Learning Interactive Real-World Simulators") [2310.06114](https://arxiv.org/abs/2310.06114) · TD-MPC2 [2310.16828](https://arxiv.org/abs/2310.16828) · Fast-WAM [2603.16666](https://arxiv.org/abs/2603.16666) · LaWAM [2606.15768](https://arxiv.org/abs/2606.15768).

Blogs and pages used: Helix https://www.figure.ai/news/helix · Genie 3 https://deepmind.google/discover/blog/genie-3-a-new-frontier-for-world-models/ · GR00T N1.5 https://research.nvidia.com/labs/gear/gr00t-n1_5/ · GR00T N1.7 README (Isaac-GR00T `main`, "What's New in GR00T N1.7").

### 0.3 Already in pyre-code (checked in `torch_judge/tasks/` and `web/src/lib/paths.json`)

The `embodied-ai` path has `vla_action_bins`, `fast_dct_action_tokens`, `act_temporal_ensemble`, `pi0_flow_matching_chunk`, `rtc_prefix_guidance`, `symlog_twohot`, `rssm_kl_balance`, `jepa_block_masks_and_loss`, `st_transformer_attention` and `world_model_cem_step`. The `vision-transformer` path ends with `noise_schedule`, `ddim_step`, `flow_matching` and `adaln_zero`. Also relevant: `kv_cache`, `moe*`, `causal_attention`, `embodied_action_stream`, `posed_frame_context`, `grpo_*`, `gae_advantage`, `ppo_*`.

**Not present** (grep over `torch_judge/tasks/*.py`): any VQ-VAE or FSQ quantizer (the only straight-through estimator is the one-hot sample inside `rssm_kl_balance`), classifier-free guidance, FiLM, diffusion policy, x-/v-prediction targets, shortcut models, per-token noise levels, and gradient stopping between model streams.

---

## 1. Candidate table

Columns: **Covered?** Y / N / partial (with task id). **Ex.** = can this be a small deterministic CPU exercise with a clear contract (H high, M medium, L low)? **Oracle** = vendorable upstream code, paper-only, or none.

### 1.1 VLA

| # | Technique | Model(s) | Source | Covered? | Ex. | Oracle | Pri |
|---|---|---|---|---|---|---|---|
| A1 | **Knowledge insulation**: action-expert queries attend to `sg(K_vlm)`, `sg(V_vlm)`. Combined loss = FAST-token CE + α·flow (α = 1). FAST tokens and action-expert tokens cannot see each other. | pi0.5 (KI checkpoints), pi*0.6 | KI paper §5.1 Eq. 4, §5.2 Eqs. 5–6, App. B. pi*0.6 §V-A ("using a stop gradient to prevent the flow-matching action expert from impacting the rest of the model"). openpi README line 8 / line 81 (`pi05_droid` "with knowledge insulation"). | N (`pi0_flow_matching_chunk` covers only the flow inputs) | H | `LR/src/lerobot/policies/pi05/modeling_pi05.py::compute_layer_complete` (line 204; KI branch lines 275–290), Apache-2.0 | **P0** |
| A2 | **RECAP advantage conditioning**: distributional value (B = 201 bins, returns normalized to (−1, 0) per task) → N-step advantage (N = 50 post-training) → binary indicator `I_t = 1(A > ε_ℓ)` as the text "Advantage: positive/negative"; indicator dropped 30% of the time; CFG at inference with β. | pi*0.6 | pi*0.6 §IV-A Eq. 1, §IV-B Eq. 3, §V-B, §V-C, §V-D, App. A-E (CFG formula, β ∈ [1.5, 2.5]), App. A-F (30% dropout, N = 50, threshold so ~30% / ~40% is positive) | N | H | **paper-only** | **P0** |
| A3 | **Latent actions**: a VQ-VAE on the frame pair `(x_t, x_{t+H})`. Its encoder serves as an inverse-dynamics model, and the quantized code is the pseudo-action for action-free video. | LAPA, UniVLA, GR00T N1 (as a "LAPA" embodiment), Genie (LAM) | GR00T N1 §2.2 "Latent Actions". LAPA paper. UniVLA paper. | N (prior doc listed `vqbet_residual_vq`, never added) | H | LAPA `laq/laq_model/nsvq.py::NSVQ.forward` (line 112; `input_data = last - first`, noise-substitution VQ, perplexity) and `replace_unused_codebooks` (line 177), MIT. UniVLA `latent_action_model/genie/modules/blocks.py::VectorQuantizer` (line 362, `cdist` + argmin + straight-through) and `ResidualVectorQuantizer` (line 416), Apache-2.0. | **P0** |
| A4 | **Parallel decoding + continuous L1 head**: K·D empty action embeddings, a bidirectional mask over them, one forward pass, an MLP head on the action hidden states, mean L1 loss | OpenVLA-OFT (26× throughput on LIBERO) | OFT §IV-A, §IV-B, App. A-A, Table II | partial (`vla_action_bins` covers the discrete head only) | M–H | OFT `prismatic/extern/hf/modeling_prismatic.py::_prepare_input_for_action_prediction` (line 734), `_process_action_masks` (line 431), `prismatic/models/action_heads.py::L1RegressionActionHead` (line 84), MIT. The bidirectional mask itself lives in a patched `transformers` fork, so spec it from §IV-B. | **P1** |
| A5 | **FiLM on ViT blocks**: `x ← (1+γ)⊙x + β` after attention and before the MLP, from the mean language embedding, shared across patches | OpenVLA-OFT+ | OFT §IV-C, App. A-C | partial (`adaln_zero` is the same algebra) | H | OFT `prismatic/models/film_vit_wrapper.py::FiLMedVisionTransformerBlock` (line 11), MIT | P2 (fold into A4 as part 2) |
| A6 | **Relative / delta EEF action chunks on SE(3)**: `T_rel = T_0^{-1} T_t` for every step of a chunk, a delta chunk relative to the previous pose, inverse back to absolute | GR00T N1.7 ("Relative EEF Action Space", a key factor in cross-embodiment transfer), LeRobot action representations | GR00T README "What's New in GR00T N1.7" item 1 | N (prior doc rejected only the joint-space version as too small) | H | Isaac-GR00T `gr00t/data/state_action/pose.py::relative_transformation` (line 58), `action_chunking.py::relative_chunking` (line 82) and `delta_chunking`, Apache-2.0 (numpy; rotation types include rot6d) | P1 |
| A7 | **Embodiment-conditioned parameters**: per-embodiment weights gathered by id, then `bmm`. Per-domain soft prompts. | GR00T N1–N1.7, X-VLA | GR00T N1 §2.1. X-VLA paper. | N | H (Easy) | Isaac-GR00T `gr00t/model/modules/embodiment_conditioned_mlp.py::CategorySpecificLinear` (line 59), `MultiEmbodimentActionEncoder` (line 177). `LR/src/lerobot/policies/xvla/soft_transformer.py::DomainAwareLinear` (line 215), soft prompt hub (line 344). | P2 |
| A8 | **adaRMSNorm time conditioning** with a residual gate (scale, shift, gate from a zero-init Dense on the time MLP) | pi0.5 | openpi `src/openpi/models/gemma.py::RMSNorm` (line 113), `_gated_residual` use (lines 311, 330), `pi0.py::embed_suffix` (lines 140–186) | partial (`adaln_zero`) | H | openpi (JAX, Apache-2.0) | P2 |
| A9 | Discrete state tokens: state digitized into 256 bins inside the text prompt | pi0.5 | openpi `src/openpi/models/tokenizer.py` lines 26, 70 | Y (`vla_action_bins`, same `np.digitize`) | — | — | skip |
| A10 | Layer skipping (first `num_vlm_layers`), expert-to-VLM layer mapping, interleaved CA/SA every n layers | SmolVLA | SmolVLA paper. `LR/src/lerobot/policies/smolvla/smolvlm_with_expert.py::get_model_layers` (line 400), `forward_cross_attn_layer` (line 294), config `configuration_smolvla.py` lines 89–97 | N | M (mostly index bookkeeping) | LeRobot, Apache-2.0 | P2 |
| A11 | FLARE: align future-observation embeddings (coef 0.2) | GR00T N1.5 | NVIDIA N1.5 page. FLARE paper 2505.15659. | N | M | **none released**: the `n1d5` branch adds `future_tokens` (`flow_matching_action_head.py` lines 196, 325) but says "not using flare now" (line 335) | P2 |
| A12 | Embodied thinking, motion transfer | Gemini Robotics 1.5 / ER 1.5 | GR 1.5 §2.1, §3.2–3.3: no loss, tokenization or architecture disclosed | N | L | none | exclude |
| A13 | S2 (7B VLM, 7–9 Hz) → one latent vector → S1 (80M, 200 Hz), with a train-time temporal offset | Helix (Figure) | Helix blog only ("a standard regression loss", no equations) | N | L | none | exclude |
| A14 | Hierarchy: a VLM emits language subcommands to pi0 | Hi Robot | Hi Robot §4.3–4.4: standard next-token CE plus pi0 flow matching, no new tensor-level algorithm | N | L | none | exclude |

### 1.2 World models

| # | Technique | Model(s) | Source | Covered? | Ex. | Oracle | Pri |
|---|---|---|---|---|---|---|---|
| B1 | **Diffusion forcing**: independent per-frame noise levels at train time; an M×T scheduling matrix (pyramid / autoregressive / full-sequence); context frames held at a small "stabilization" noise level | Diffusion Forcing, Oasis, Dreamer 4 (per-step τ, d ∈ [0,1]^T; τ_ctx = 0.1), GameNGen (context noise augmentation: max 0.7, 10 buckets), Self Forcing (`context_noise`) | DF §3.2 (Alg. 1–2), §3.3. Dreamer 4 §3.2. GameNGen §3.2.1, §4.2. | N | H | DF `algorithms/diffusion_forcing/df_base.py::_generate_noise_levels` (line 172), `_generate_scheduling_matrix` / `_generate_pyramid_scheduling_matrix` (lines 188, 199), `models/diffusion.py` stabilization (`stabilization_level`, lines 42, 383+), MIT. open-oasis `generate.py` lines 51–125 (`stabilization_level = 15`, v-prediction DDIM, sliding window), MIT. | **P0** |
| B2 | **Shortcut forcing**: x-prediction shortcut model; `d ~ 1/U{1,2,4,…,K_max}`, τ on the d-grid; bootstrap target `sg(b'+b'')/2` from two half steps; `(1−τ)²` scale; ramp weight `w(τ) = 0.9τ + 0.1`; K = 4 steps at inference | Dreamer 4 | Dreamer 4 §2 Eq. 4, §3.2 Eqs. 6–8 | N | H | **paper-only** (no official code) | **P1** |
| B3 | **Self-forcing rollout**: block-autoregressive few-step generation at train time; per block, a random "exit" denoising step is the only one with gradient; the cache is refreshed by re-running the output at `context_noise` | Self Forcing (CausVid lineage), real-time interactive video WMs | Self Forcing paper. Code below. | N | M–H (toy generator; DMD loss excluded) | Self-Forcing `pipeline/self_forcing_training.py::generate_and_sync_list` (line 40), `inference_with_trajectory` (lines 60–215), Apache-2.0 | **P1** |
| B4 | **Clean/noisy teacher-forcing mask**: the sequence holds clean and noisy copies; clean→clean block-causal, noisy→clean strictly earlier frames, noisy→noisy same frame only, plus sequence and window masks | LingBot-VA (world action model); same pattern as CausVid / Self Forcing teacher forcing | `LR/src/lerobot/policies/lingbot_va/utils.py::_get_mask_mod` (line 434), `init_mask` (line 368) | partial (`st_transformer_attention` has a plain causal-in-time mask) | H | LeRobot, Apache-2.0 (header: Robbyant Team + HF) | P1 |
| B5 | **Action-conditioned latent rollout loss**: teacher-forced one-step loss + autoregressive rollout loss (`auto_steps = 2`), both `mean(|z−h|^p)/p` on layer-normed targets; frame-block causal mask with action/state tokens | V-JEPA 2-AC | V-JEPA 2 paper (AC section). `app/vjepa_droid/train.py` lines 417–449. `configs/train/vitg16/droid-256px-8f.yaml` (`auto_steps: 2`, `loss_exp: 1.0`). `src/models/utils/modules.py::build_action_block_causal_attention_mask` (line 12). | partial (`jepa_block_masks_and_loss` has the loss form; `world_model_cem_step` has planning) | H | vjepa2, MIT | P1 |
| B6 | **FSQ**: bound with tanh (even-level offset), round with STE, normalize by `L//2`, mixed-radix `codes ↔ indices` | Cosmos discrete tokenizers (levels `[8,8,8,5,5,5]`); FSQ | FSQ paper (App. A.1 JAX code). Cosmos-Tokenizer `cosmos_tokenizer/modules/quantizers.py::FSQuantizer` (line 71; `bound` line 136; `quantize` line 143; `codes_to_indices`, `indices_to_codes`), `networks/configs.py` line 89 | N | H | Cosmos-Tokenizer, Apache-2.0 (adapted from lucidrains at `9502a1f`); lucidrains `finite_scalar_quantization.py` (`round_ste` line 52, `bound` line 147), MIT | **P1** |
| B7 | PMPO sign-only policy loss in imagination: `D+ / D−` by sign of `A = R^λ − v`; α = 0.5; KL to a frozen prior (β = 0.3) | Dreamer 4 | Dreamer 4 §3.3 Eqs. 10–11 | N | H | paper-only | P2 (overlaps the RL paths) |
| B8 | TD-MPC2 pieces: SimNorm (softmax over groups of 8); latent consistency loss weighted `ρ^t` (ρ = 0.5); Q-ensemble of 5, min over 2 random members | TD-MPC2 | `tdmpc2/common/layers.py::SimNorm` (line 74), `common/world_model.py::Q` (line 186; `randperm(num_q)[:2]` line 212), `tdmpc2.py` losses (lines 225–303), `config.yaml` lines 21, 62, 64 | partial (`symlog_twohot`, `world_model_cem_step` tdmpc mode) | H | MIT | P2 |
| B9 | EDM preconditioning | DIAMOND | prior doc §3.2 table | N | H | DIAMOND, MIT | P2 (unchanged from prior doc) |
| B10 | MaskGIT schedule for token world models | Genie | prior doc §3.2 table | N | H | open-genie (unofficial) | P2 (unchanged) |
| B11 | Context-frame CFG (guidance on past observations only, weight 1.5) | GameNGen | GameNGen §3.3.2 | N | H | paper-only | fold into A2's CFG part or B1 |
| B12 | CDiT (conditional DiT over past frames, adaLN on time/action) | Navigation World Models | NWM paper | partial (`adaln_zero`, `st_transformer_attention`) | M | **CC BY-NC**, do not vendor | exclude |
| B13 | Genie 2 / Genie 3 | DeepMind | Genie 3 blog: autoregressive, 720p, 24 fps, ~1 min memory; "does not disclose architecture" | N | L | none | exclude |
| B14 | Cosmos-Predict2.5 / Cosmos-Reason1 / UniSim | NVIDIA, Google | papers above | N | L | — | exclude (scale and data; the tokenizer part is B6) |
| B15 | World action models (video + action co-generation; Fast-WAM skips future imagination at test time) | LingBot-VA, Fast-WAM, LaWAM | LeRobot docs `docs/source/{lingbot_va,fastwam,lawam}.mdx` | partial via B4 | L as a whole | LeRobot | watch; B4 is the teachable piece |

---

## 2. Recommended new exercises (top 8)

Position = where each sits in the existing `embodied-ai` path (`web/src/lib/paths.json`). Adding 8 takes the path to 18 problems. If that is too long, put A1/A2/A3/A4 and B1/B2/B3/B6 into a second path, "Embodied AI II: frontier VLAs and generative world models", with `prerequisites: ["embodied-ai"]`. The paths page uses group cards, and a new path in `architectures` does not list anything twice.

**1. `oft_parallel_action_decoding`** (VLA, Medium; after `vla_action_bins`; P1)
Contract: `build_parallel_inputs(prompt_emb (B,P,E), K, D, action_emb) -> (inputs (B,P+K·D+1,E), attn_mask (S,S) bool, action_pos)`. The prompt part is causal, every action placeholder sees the prompt and all placeholders (bidirectional block), then a stop token. Also `l1_chunk_loss(hidden (B,K·D,E), head, target (B,K,D))`: reshape the K·D action hidden states into K rows, apply the head, and take the mean L1. The test checks that perturbing a later placeholder changes earlier action outputs (bidirectional) but never the prompt outputs.
Oracle: OFT `modeling_prismatic.py` lines 734–760, `action_heads.py` line 84 (MIT); mask semantics from OFT §IV-B. Optional part 2: FiLM `(1+γ)x+β` (A5).

**2. `latent_action_quantizer`** (VLA, Medium; after `fast_dct_action_tokens`, as "actions for action-free video"; P0)
Contract: `latent_action(f_t, f_tH, codebook, mode, noise=None) -> (z_q, idx, perplexity)`. The input is the feature difference `f_tH − f_t` (LAPA). `mode="ste"` returns `x + sg(q − x)` (UniVLA, Genie). `mode="nsvq"` returns `x + (‖x−q‖/‖n‖ + eps)·n` with the noise `n` passed in. Also `replace_dead_codes(codebook, usage, threshold, donors)` with deterministic donors. Tests check gradients (codebook gets none under STE; the encoder gets the identity), the perplexity formula, and tie-breaking in argmin.
Oracle: LAPA `nsvq.py` lines 112–230 (MIT), UniVLA `blocks.py` lines 362–440 (Apache-2.0).

**3. `knowledge_insulation_attention`** (VLA, Hard; after `pi0_flow_matching_chunk`; P0)
Contract: `ki_joint_attention(q, k, v, prefix_len, mask, insulate) -> out`. Prefix queries attend normally. Suffix (action-expert) queries attend to `cat[sg(k[:prefix]), k[prefix:]]` and the same for v. Forward values must equal plain joint attention. Plus `ki_mask(n_vlm, n_fast, n_act)`, where FAST and action tokens are mutually invisible, and `co_vla_loss(ce_fast, fm, m_lang, m_act, alpha=1.0)`. Autograd tests: the flow loss gives zero gradient to the VLM's `W_k`/`W_v`, while the FAST CE still reaches them.
Oracle: `LR/.../pi05/modeling_pi05.py` lines 204–290 (Apache-2.0); KI paper §5.1–5.2, App. B.

**4. `recap_advantage_conditioning`** (VLA, Hard; capstone after `rtc_prefix_guidance`; P0)
Contract: `recap_labels(rewards (E,T), values_probs (E,T+1,B), bin_centers, n_step, eps_quantile, drop_mask) -> (adv, indicator_tokens)`. Value = Σ p·v(b). Advantage = n-step return + V(t+N) − V(t), truncated at the episode end. The threshold ε is the per-task quantile so that a given fraction is positive. Dropped rows get a "no indicator" id. Also `cfg_velocity(v_uncond, v_cond, beta) = v_u + β(v_c − v_u)`. Tests: β = 1 equals the conditional model, β = 0 equals the unconditional one, and human-correction rows are forced positive.
Oracle: **paper-only** (pi*0.6 §IV, §V-B to §V-D, App. A-E, A-F). This is allowed by the prior doc's decision 1. One ambiguity: §V-D says the threshold is a percentile of *values*, while App. A-F describes it as a fraction of *data with positive advantage*. The exercise should follow A-F and say so.

**5. `fsq_quantizer`** (World Models, Medium; before `st_transformer_attention`, as the tokenizer step; P1)
Contract: `fsq(z (...,d), levels) -> (zhat, indices)` with `bound` (tanh, `half_l = (L−1)(1+eps)/2`, offset 0.5 for even L), STE round, divide by `L//2`. Also `indices_to_codes(indices, levels)` via the basis `cumprod([1]+levels[:-1])`. Tests: the round trip is exact over the full implicit codebook; levels `[8,8,8,5,5,5]` give a codebook of 64 000; the gradient is the identity through the round. Mutations: no even-level offset; the wrong basis order.
Oracle: Cosmos-Tokenizer `quantizers.py` lines 71–175 (Apache-2.0), cross-checked against lucidrains (MIT).

**6. `diffusion_forcing_schedule`** (World Models, Medium; after `st_transformer_attention`; P0)
Contract: `train_noise_levels(T, B, K, gen, frame_mask)` (independent per frame; missing frames forced to full noise). `scheduling_matrix(horizon, K, kind, uncertainty_scale)` for `pyramid | autoregressive | full_sequence | trapezoid`, clipped to [0, K]. `context_levels(n_ctx, stabilization_level)`. Tests compare against hand-built matrices (row 0 is all K, the last row is all 0, each column is monotone non-increasing). Mutation: `int(t*u)` placed outside the subtraction.
Oracle: DF `df_base.py` lines 172–215 (MIT), cross-checked with open-oasis `generate.py` (MIT). The docs connect Dreamer 4's τ_ctx = 0.1 and GameNGen's noise augmentation (§3.2.1) to the same idea.

**7. `shortcut_forcing_loss`** (World Models, Hard; after `diffusion_forcing_schedule`; P1)
Contract: `shortcut_forcing_loss(f, z1, z0, tau, d, d_min, action) -> loss`, where `f` is a provided deterministic toy x-predictor. If `d == d_min`: `w(τ)·‖f(z̃,τ,d)−z1‖²`. Otherwise two half steps produce `b'` and `b''`, and the loss is `w(τ)(1−τ)²‖(ẑ1−z̃)/(1−τ) − sg((b'+b''))/2‖²`, with `z̃ = (1−τ)z0 + τ z1` and `w = 0.9τ + 0.1`. Part 2: `sample_shortcut_grid(K_max)` checks that τ lies on the d-grid. The test verifies that no gradient flows through the bootstrap target.
Oracle: **paper-only** (Dreamer 4 Eqs. 4, 6–8). The spec must pin down whether `w(τ)` multiplies both branches; the paper introduces it in the text, outside Eq. 7.

**8. `self_forcing_rollout`** (World Models, Hard; capstone after `shortcut_forcing_loss`; P1)
Contract: `self_forcing_rollout(gen, noise (B,F,C), step_list, frames_per_block, exit_steps, context_noise, add_noise) -> output`. For each block, iterate `step_list` under `no_grad` until `exit_steps[block]`, run that step with gradient, then re-run the clean output at `context_noise` to refresh a toy KV cache (a list the generator appends to). Tests: gradient reaches only the exit-step calls (count with hooks); `last_step_only` gives the plain few-step sampler; the cache length grows by one entry per block.
Oracle: Self-Forcing `self_forcing_training.py` lines 40–215 (Apache-2.0). Distributed sync is removed and `exit_steps` is passed in.

**Alternates (P1, if any of the above slips):** `vjepa2_ac_rollout_loss` (B5), `relative_eef_chunk` (A6), `clean_noisy_teacher_mask` (B4).

---

## 3. Deliberately excluded

- **Gemini Robotics 1.5 / ER 1.5 (embodied thinking, motion transfer).** The paper discloses no loss, tokenization or decoder (§2.1, §3.2–3.3). Thinking is "text before actions", which the Latent Reasoning path and ordinary SFT masks already cover.
- **Helix (Figure).** Blog only. "A standard regression loss" with no equations or code. The S1/S2 rate split is a systems topic and overlaps `embodied_action_stream`.
- **Hi Robot.** By its own §4.3–4.4 the contribution is the hierarchy and synthetic data, with next-token CE plus pi0 flow matching. No new tensor primitive.
- **Genie 2 / Genie 3.** No paper or code; the Genie 3 blog discloses only resolution, fps and memory horizon. Genie (v1) is already covered by `st_transformer_attention`, and A3 now covers its latent-action part.
- **Navigation World Models (CDiT).** The repo is CC BY-NC 4.0, so it cannot be vendored. The technique is adaLN-conditioned DiT attention, already covered by `adaln_zero` and `st_transformer_attention`.
- **Cosmos-Predict2.5, Cosmos-Reason1, UniSim, GR00T N1.7's EgoScale pretraining.** The gains come from scale and data. The one teachable Cosmos piece (FSQ) is #5.
- **FLARE (GR00T N1.5).** No released loss code (`n1d5` `flow_matching_action_head.py` line 335, "not using flare now"). Revisit if NVIDIA releases it.
- **DMD distillation in Self Forcing.** It needs teacher and critic score networks, which is too big for one exercise. #8 keeps only the rollout and gradient-truncation logic.
- **SmolVLA async inference.** Excluded in the prior doc (overlaps `embodied_action_stream`). Layer skipping (A10) is P2 bookkeeping.
- **pi0.5 discrete state tokens** are already taught by `vla_action_bins`. **adaRMS** (A8) and **FiLM** (A5) are near-duplicates of `adaln_zero`; use them only as part 2 of another exercise.

---

## 4. Risks and open questions

1. **Two P0/P1 items are paper-only** (RECAP #4, shortcut forcing #7). Decision 1 in the prior doc allows this, but both papers leave a detail open (RECAP's threshold definition; where Dreamer 4's ramp weight applies). Each exercise must state its choice and cite the section.
2. **The Self-Forcing oracle is CUDA- and distributed-flavored** (`.cuda()`, `dist.broadcast`). The vendored copy has to strip these, which makes it partly author-written.
3. **The OFT bidirectional mask is not in the OFT repo.** It comes from a patched `transformers` install per the README, so #1's mask is specified from §IV-B.
4. **LeRobot moves fast.** All LeRobot citations were checked at `75cd2ba` and still match `HEAD` `24f15b9` on 2026-10-12.
5. **Not verified:** the content of FLARE paper 2505.15659 (title only); the X-VLA paper body (only the LeRobot port was read); the Cosmos-Predict2.5 body; and RECAP's CFG implementation detail beyond App. A-E's gradient formula.
