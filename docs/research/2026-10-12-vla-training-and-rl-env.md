# Research: how VLA training differs from LLM training, and a Robot RL Environment path

Date: 2026-10-12
Status: Research input for curriculum design. No exercises were created.
Companion to: `docs/research/2026-10-12-vla-world-models-latent-reasoning.md` (pinned LeRobot, openpi, OpenVLA, ACT sources; not repeated here).

How this was researched: upstream repositories were shallow-cloned on 2026-10-12 and read directly; commit hashes are the `HEAD` of those clones. Licenses come from each repo's `LICENSE` file and the GitHub API (`repos/<r>.license.spdx_id`). Paper claims were read from `arxiv.org/abs/<id>` and `arxiv.org/html/<id>` pages; section numbers are the paper's own. Anything not verified this way is marked **Unverified**.

---

## 0. Pinned upstream sources

| Short name | Repo | Commit (date) | License |
|---|---|---|---|
| Gymnasium 1.4.0 | https://github.com/Farama-Foundation/Gymnasium | `d77197c284406e6094c9d1a74d4988a8749ce849` (2026-10-10) | MIT (`LICENSE`) |
| ManiSkill3 | https://github.com/haosulab/ManiSkill | `62ff3a5896b4d5b4cf0ac4c8d79afe600c9404a3` (2026-08-01) | Apache-2.0 (`LICENSE`) |
| Isaac Lab | https://github.com/isaac-sim/IsaacLab | `b0542fe2d45bf91c4e1d9ef6952b9c709c80b4e8` (main, 2026-07-24; one file fetched raw) | BSD-3-Clause, parts Apache-2.0 (project docs) |
| RLinf | https://github.com/RLinf/RLinf | `0e9438dee0088be7673a1fb4789018aea168db1a` (2026-10-12) | Apache-2.0 (`LICENSE`, API) |
| SimpleVLA-RL | https://github.com/PRIME-RL/SimpleVLA-RL | `7c51662df27b586f9e8a1ab35fcf849f2b8852f9` (2026-01-06) | MIT (`LICENSE`, API) |
| RL4VLA | https://github.com/gen-robot/RL4VLA | `de46e0174e7ac310255a2836718fcefab28b8963` (2025-08-25) | MIT (`LICENSE`, API) |
| VLA-RL | https://github.com/GuanxingLu/vlarl | `212379e840376e07505cba398c0c587d96a13678` (2025-09-13) | Apache-2.0 (API) |
| openpi | https://github.com/Physical-Intelligence/openpi | `215abfb217dbac7d5f1273282331b9b1866c0479` (2026-08-24) | Apache-2.0 (`LICENSE`) |
| DSRL | https://github.com/ajwagen/dsrl | `8d21b9cf55459f022c0e769bca1343edd9d30e5a` (2025-08-05) | **No license** (no `LICENSE` file; API `none`) — cite only, never vendor |
| RIPT-VLA | https://github.com/Ariostgx/ript-vla | `440990e8864e12e4578b490ff6359e4f2c49ae3e` (2025-06-04) | **No license** (API `none`) — cite only |

Papers (titles verified on arXiv): Open X-Embodiment [2310.08864](https://arxiv.org/abs/2310.08864), Octo [2405.12213](https://arxiv.org/abs/2405.12213), OpenVLA-OFT "Fine-Tuning Vision-Language-Action Models: Optimizing Speed and Success" [2502.19645](https://arxiv.org/abs/2502.19645), pi0.5 [2504.16054](https://arxiv.org/abs/2504.16054), Knowledge Insulating VLAs [2505.23705](https://arxiv.org/abs/2505.23705), LIBERO [2306.03310](https://arxiv.org/abs/2306.03310), SIMPLER "Evaluating Real-World Robot Manipulation Policies in Simulation" [2405.05941](https://arxiv.org/abs/2405.05941), ManiSkill3 [2410.00425](https://arxiv.org/abs/2410.00425), Isaac Lab [2511.04831](https://arxiv.org/abs/2511.04831), SimpleVLA-RL [2509.09674](https://arxiv.org/abs/2509.09674), RL4VLA "What Can RL Bring to VLA Generalization? An Empirical Study" [2505.19789](https://arxiv.org/abs/2505.19789), VLA-RL [2505.18719](https://arxiv.org/abs/2505.18719), RIPT-VLA "Interactive Post-Training for Vision-Language-Action Models" [2505.17016](https://arxiv.org/abs/2505.17016), pi\*0.6 / RECAP [2511.14759](https://arxiv.org/abs/2511.14759), ConRFT [2502.05450](https://arxiv.org/abs/2502.05450), PA-RL "Policy Agnostic RL" [2412.06685](https://arxiv.org/abs/2412.06685), Policy Decorator [2412.13630](https://arxiv.org/abs/2412.13630), DSRL "Steering Your Diffusion Policy with Latent Space RL" [2506.15799](https://arxiv.org/abs/2506.15799), ReinFlow [2505.22094](https://arxiv.org/abs/2505.22094), Flow-GRPO [2505.05470](https://arxiv.org/abs/2505.05470), pi_RL [2510.25889](https://arxiv.org/abs/2510.25889), RLinf-VLA [2510.06710](https://arxiv.org/abs/2510.06710).

Path shorthand below: `GYM/` = Gymnasium, `MS/` = ManiSkill, `RLINF/` = RLinf, `SVR/` = SimpleVLA-RL, `OPI/` = openpi, at the commits above.

---

## 1. VLA vs LLM training at a glance

| Stage / concern | LLM | VLA | Source |
|---|---|---|---|
| Pretraining data | Web text, one modality, one "embodiment" | Robot trajectories from many robots: OXE pools 60 datasets from 34 labs, 1M+ trajectories, 22 embodiments, stored as RLDS | OXE §III-A |
| Output space | Shared vocabulary | Per-robot continuous actions with different meaning (absolute vs delta, velocity vs position, frames not aligned) | OXE §IV-A |
| Normalization | None (tokens) | Statistics **per dataset/embodiment**: OXE normalizes each dataset's actions before discretization; openpi uses z-score for pi0 and q01/q99 quantile (`(x-q01)/(q99-q01+1e-6)*2-1`) for other models | OXE §IV-A; `OPI/src/openpi/transforms.py` L137–181, `training/config.py` L187 |
| Heterogeneous dims | n/a | Pad to a fixed width: openpi `action_dim = 32`, `pad_to_dim`; quantile unnormalize passes padded dims through; Octo zero-pads missing cameras and fully masks tokens of absent observations; new embodiment = new action head (Octo ALOHA, 14-D) | `OPI/src/openpi/models/pi0_config.py` L25; `transforms.py` L175–181, L423; Octo §III-B, App. F-C |
| Action representation | n/a | Absolute ↔ delta per masked dim (`DeltaActions`/`AbsoluteActions`); gripper conventions differ ([0,1] vs [-1,1], sign flips) | `OPI/src/openpi/transforms.py` L204–238; `SVR/verl/utils/libero_utils.py` L156, L196 |
| Observation | Token context that grows | Multi-camera images + proprio state, short history (Octo: 2 frames); openpi LIBERO feeds agent view + wrist view + 8-D state each query; no growing context | Octo §III-D; `OPI/examples/libero/main.py` L117–141 |
| Output granularity | One token per step | **Action chunks**: pi0 `action_horizon = 50`; Octo bimanual chunk 64 executing 12; OpenVLA-OFT parallel decoding + chunking + L1 regression (26x throughput) | `pi0_config.py` L26; Octo App. F-C; OFT abstract |
| Control rate | Latency only matters to users | Hard real-time: RT-1-X/RT-2-X run at 3–10 Hz; pi0.5 commands at 50 Hz with chunking; Isaac Lab `decimation` = physics steps per policy step | OXE §IV-C; pi0.5 §IV-E; Isaac Lab `envs` API docs |
| Co-training | Mid-training data mixes | VLM data kept in the mix to preserve semantics: pi0.5 mixes mobile/multi-env/cross-embodiment robot data, high-level subtask labels and web captioning/VQA; pre-train 280k steps on FAST tokens (α=0), post-train 80k steps adding a flow-matching expert (α=10) | pi0.5 §IV-B, §IV-C, §IV-D |
| Head/backbone interaction | n/a | **Knowledge insulation**: stop-gradient from the flow-matching action expert into the VLM; backbone still learns from FAST discrete-token loss; pi0 needed ~7.5x the steps | KI §5.1, §5.2, §6 |
| Mixture weights | Data-size heuristics | Octo: start from dataset size, double diverse datasets, downweight repetitive ones | Octo §III-B, App. C |
| Evaluation | Static benchmarks, pass@k | Closed-loop rollouts with success flags: LIBERO (4 suites, 130 tasks; fixed benchmark initial states per episode); SIMPLER sim evals chosen for **rank correlation** with real (MMRV, Pearson); real-world success rate | LIBERO abstract; `OPI/examples/libero/main.py` L97; SIMPLER §III-A, §III-B |
| Eval protocol details | Prompt + decode | openpi LIBERO: 10 no-op "settle" steps, replan every 5 of 50 predicted actions, per-suite step limits (220–520), success = `done` from the env | `OPI/examples/libero/main.py` L29, L37, L61–69, L104–160 |
| RL reward | Verifier on text | Usually sparse binary success from simulator state (`SVR` `RobRewardManager.verify` = `complete`); ManiSkill offers sparse/dense/normalized_dense | `SVR/verl/trainer/main_ppo.py` L38; `MS/mani_skill/envs/sapien_env.py` L649–691 |
| RL rollout cost | Cheap generation, tools optional | A physics simulator + renderer in the loop for hundreds of steps, each step a full VLA forward; GPU-parallel sims (ManiSkill3 30k+ FPS, Isaac Lab) | ManiSkill3 abstract; Isaac Lab abstract |

---

## 2. RL for VLAs: what the primary sources do

### 2.1 On-policy policy gradient on discrete action tokens (closest to LLM RL)
- **SimpleVLA-RL** (OpenVLA-OFT, veRL-based). Reward: "trajectory-level rewards of either 0 or 1 based solely on task completion", "uniformly propagated to the individual action tokens" (§3.2). GRPO without KL "following DAPO" (§3.4), clip [0.8, 1.28] (§3.3, §4.1), dynamic sampling drops all-success/all-fail groups (§3.3), rollout temperature 1.6, greedy eval (§4.1); chunk length 8 (LIBERO) / 25 (RoboTwin) (§4.1); "pushcut" emergent pushing behavior (§6.1).
  Code: the env worker executes a chunk and **breaks mid-chunk** on `done or finish_step >= max_steps` (`SVR/verl/workers/rollout/rob_rollout.py` L396–410); the token mask is a prefix `arange(T) < finish_step * action_token_len` (`SVR/verl/trainer/ppo/ray_trainer.py` L128–131); GRPO advantage per group index (`SVR/verl/trainer/ppo/core_algos.py` L173).
- **RL4VLA** (OpenVLA on ManiSkill pick-and-place, WidowX). PPO beats GRPO and DPO (§4.1); hypothesis: GRPO advantages destabilized by non-stationary POMDP dynamics. Shared actor-critic backbone, 3-layer MLP value head on the hidden state at **the first action-token position** (§4.2); 1 PPO epoch (§4.2); action probability = product of action-token probabilities (§A.1); reward 0.1 for holding the right object, 1.0 for placing (§5.1). RL ≈ SFT on vision, better on semantics, much better on execution (§5.3).
- **VLA-RL** treats a trajectory as a "multi-modal multi-turn conversation", adds a VLM-based process reward model for dense reward, critic warmup, GPU-balanced vectorized envs (abstract). Code: GAE with `nextnonterminal = 1 - dones[t+1]` (`ppo_vllm_ray_fsdp_v3.py` L1549–1558).
- **RIPT-VLA**: sparse binary success, dynamic rollout sampling and leave-one-out (RLOO) advantages (abstract). Code: a group is `rloo_batch_size` rollouts **from the same initial state** (`np.tile(env_init_state, ...)`), and init states whose group was all-success are skipped (`ript/algos/rl_optimizers/rollout_generator.py` L202–238, L269–271). No license.

### 2.2 Continuous / flow-matching heads: the log-likelihood problem
- Flow matching has no tractable action likelihood; pi\*0.6 says policy gradient is "difficult to apply to flow matching models" and PPO needed a tiny trust region and still underperformed (§IV-B, §VI-C3).
- **Flow-GRPO** (image generation): ODE→SDE conversion with the same marginals, fewer denoising steps in training (abstract).
- **ReinFlow**: learnable noise per denoising step turns the flow into a Gaussian Markov chain; the chain log-likelihood = log N(x0) + Σ per-step Gaussian log-densities (§4.1, Eq. 6–7); PPO + GAE (Alg. 2).
- **pi_RL** (pi0, pi0.5 in RLinf): Flow-Noise (learned noise net, exact chain log-prob, Eq. 5) and Flow-SDE (two-layer MDP: denoising inner loop inside env outer loop; one random SDE step per action, rest ODE) (§4.1–4.2.3). A chunk of H actions is one macro-step with reward = sum over the chunk (§4.3.1). PPO with GAE, critic on the VLM output for pi0.5 (§4.3.2). Few-shot LIBERO: pi0 57.6% → 97.6%, pi0.5 77.1% → 98.3% (App. C.1, Tab. 3).
  Code: `RLINF/rlinf/models/embodiment/openpi/sampling/rl_sampler.py` `sample_mean_var` (flow_ode / flow_sde / flow_cps / flow_noise) and `gaussian_logprob` (zero where `sigma == 0`).
- **Granularity** is an explicit switch in RLinf: `logprob_type` ∈ `token_level` (per action dim), `action_level` (sum over dims), `chunk_level` (sum over chunk and dims), and `reward_type == "chunk_level"` sums rewards over the chunk (`RLINF/rlinf/algorithms/utils.py` L69–90, L282–380).

### 2.3 Avoiding the likelihood altogether
- **RECAP / pi\*0.6**: distributional value (201 bins) trained on Monte Carlo returns; reward −1 per step, 0 on success, large negative on failure, so value ≈ −steps-to-go (§IV-A, Eq. 5, §V-C); binarized improvement indicator `I_t = 1(A > ε)` with ε at the 30th percentile of value predictions (§IV-B, §V-D); fed as text "Advantage: positive/negative" before the actions, randomly dropped so CFG is possible (§V-B, §VI-B); human corrections always labeled positive (§IV-B).
- **DSRL**: RL over the diffusion policy's **input noise**; the frozen policy becomes part of the environment (§4.1); DSRL-SAC and noise-aliased DSRL-NA (§4.2, §5.6); latent box `[-b, b]` (App. B); pi0 on a LIBERO task ~20% → ~100% after ~10k online samples (§5.5). Its `ActionChunkWrapper` **keeps stepping after `done` inside a chunk** and sums all rewards (`dsrl/env_utils.py` L161–184) — a counter-example to the mid-chunk stop used elsewhere.
- **Residual RL — Policy Decorator**: `a = π_base(s) + π_res(s)`, base frozen, SAC, residual bounded by `α·tanh`, behavior uses the residual with probability ε ramped 0→1 over H steps (§4.1, §4.2).
- **PA-RL**: sample candidates from the base policy, re-rank by Q, gradient-refine, then distill with a supervised loss; works for diffusion and autoregressive policies; OpenVLA real-world 40% → 70% in 40 min (abstract).
- **ConRFT**: offline BC + Q-learning, then online consistency-policy fine-tuning with human interventions; 96.3% over 8 real tasks in 45–90 min (abstract).

### 2.4 Credit assignment and grouping, summarized
- LLM GRPO groups by prompt; VLA GRPO/RLOO groups by **(task, initial state)** (RIPT code; RLinf `use_fixed_reset_state_ids`, `RLINF/rlinf/envs/sim/maniskill/maniskill_env.py` L382).
- Step-level (per env step) vs chunk-level (macro-step) rewards/advantages are both used; pi_RL and RLinf `chunk_level` treat the chunk as one decision.
- PPO in sims (RL4VLA, pi_RL, VLA-RL) uses a critic; GRPO (SimpleVLA-RL) relies on binary outcome + dynamic sampling.

---

## 3. RL environment design: robots vs LLM agents

1. **terminated vs truncated.** Gymnasium: terminated = terminal state of the MDP (positive or negative); truncated = condition outside the MDP, typically a time limit (`GYM/gymnasium/core.py` L100–106). `TimeLimit` sets `truncated=True` when `elapsed >= max_episode_steps` (`GYM/gymnasium/wrappers/common.py` L130–136). In robot sims success usually sets `terminated` (ManiSkill: `terminated = success | fail`, `MS/mani_skill/envs/sapien_env.py` L1062–1071; LIBERO `done` = success). Value targets must **bootstrap through truncation but not termination**: ManiSkill PPO computes `V(final_observation)` for done envs and uses it as the next value (`MS/examples/baselines/ppo/ppo.py` L331, L345). RLinf's GAE takes a single `dones` flag and documents "does not support auto-reset" (`RLINF/rlinf/algorithms/advantages.py` L25–80).
2. **Autoreset modes (Gymnasium 1.x).** `AutoresetMode.NEXT_STEP` (default for Sync/Async vector envs and `make_vec`), `SAME_STEP`, `DISABLED` (`GYM/gymnasium/vector/vector_env.py` L41–46; `sync_vector_env.py` L84). NEXT_STEP: the step *after* a done ignores the action and returns the reset obs with reward 0 and both flags False (`sync_vector_env.py` L293–299) — training must drop or mask that transition (Farama "Vector Autoreset Mode" post). SAME_STEP: reset inside the same call; the true last obs is `info["final_obs"]`, last info `info["final_info"]` (`sync_vector_env.py` L318–334). ManiSkill hard-codes SAME_STEP and calls it `final_observation` (`MS/mani_skill/vector/wrappers/gymnasium.py` L66–68, L164–175). Isaac Lab resets done envs inside `step` and recomputes observations **after** the reset ("done after reset to get the correct observations for reset envs"), so the pre-reset observation is not returned (`isaaclab/envs/manager_based_rl_env.py`, `step`, read at the commit above).
3. **Vectorized / GPU-parallel envs and partial reset.** ManiSkill resets a subset via `options={"env_idx": ...}`; partial reset is incompatible with reconfiguration (`MS/.../gymnasium.py` L75–78, `sapien_env.py` L904–907). `ignore_terminations=True` runs every env for the full horizon and reports `success_at_end` (`gymnasium.py` L150–155); the PPO baseline trains with partial resets and evaluates without (`MS/examples/baselines/ppo/ppo.py` L60–62, L212–213).
4. **Action chunk execution inside one env step.** Three upstream behaviors:
   - stop at the first done, discard later actions: ManiSkill `ActionRepeatWrapper` (only not-done envs' obs/reward/info are updated, `MS/mani_skill/utils/wrappers/action_repeat.py`), SimpleVLA-RL env worker (`break`);
   - step all actions but record a **valid-action prefix mask** including the ending action, fold termination/truncation onto the chunk's last index, and autoreset after the chunk: RLinf LIBERO `chunk_step` (`RLINF/rlinf/envs/sim/libero/libero_env.py` L933–1030, `RLINF/rlinf/envs/utils.py` L134 `valid_action_mask_from_counts`);
   - step everything and OR the dones (DSRL; RLinf ManiSkill `chunk_step` without the mask, `maniskill_env.py` L327–375).
   Isaac Lab's `decimation` is the physics-side analogue: one policy action held for `decimation` sim steps (`manager_based_rl_env.py` `step` loop).
5. **Observation dicts.** Images from several cameras + proprio state; LIBERO images are rotated 180° to match training preprocessing and resized with padding (`OPI/examples/libero/main.py` L114–122). RLinf extracts `full_image`, `wrist_image`, state (`libero_env.py` L756+).
6. **Reset, seeds, initial-state distributions.** Gymnasium: call `reset(seed=...)` once, then unseeded; a seeded reset must be deterministic (`GYM/gymnasium/core.py` reset docstring; `GYM/gymnasium/utils/env_checker.py` L79 `check_reset_seed_determinism`). ManiSkill keeps a main RNG and per-episode RNG; seeds default to 2022…2022+num_envs−1; a list of seeds seeds envs individually (`sapien_env.py` L861–917). PickCube samples cube pose/goal in `_initialize_episode` (`MS/mani_skill/envs/tasks/tabletop/pick_cube.py` L106–130). LIBERO and RIPT use fixed benchmark init states (`env.set_init_state(initial_states[i])`).
7. **Success detection.** Task `evaluate()` returns `success` from simulator state (PickCube: object within `goal_thresh` **and** robot static, L147–159); dense reward is a shaped sum normalized by 5 for `normalized_dense` (L161–191).
8. **Episode statistics.** Gymnasium `RecordEpisodeStatistics` adds `info["episode"] = {r, l, t}` on done (`GYM/gymnasium/wrappers/common.py` L468+). ManiSkill `record_metrics` tracks `success_once`, `fail_once`, `return`, `episode_len`, plus `success_at_end` under `ignore_terminations`, and clears them per `env_idx` on partial reset (`MS/.../gymnasium.py` L80–157). RLinf LIBERO accumulates return only until first success and records the success episode length (`libero_env.py` L732–756).
9. **Wrapper stack.** Gymnasium ships `TimeLimit`, `ClipAction`, `RescaleAction`, `NormalizeObservation`, `RecordEpisodeStatistics`, `RecordVideo` (`GYM/gymnasium/wrappers/`). On the VLA side the "action wrapper" is the unnormalize → absolute → gripper-convention chain (`OPI/src/openpi/transforms.py`; `SVR/verl/utils/libero_utils.py` `normalize_gripper_action` binarizes with `np.sign`, so 0.5 maps to 0, not ±1).
10. **Domain randomization.** Isaac Lab: event terms for randomization and actuator models (Isaac Lab abstract); ManiSkill reconfiguration (`reconfiguration_freq`) reloads assets, per-episode randomization lives in `_initialize_episode`. SIMPLER found visual matching beats randomized "variant aggregation" for sim-real correlation (§IV-B, §VI-B). **Unverified**: Isaac Lab `EventTerm` mode names (`startup`/`reset`/`interval`) were not read from source.

---

## 4. Do the existing pyre-code exercises transfer to VLA RL?

**Key structural difference.** An LLM agent rollout is *one growing token sequence* in which policy tokens and environment tokens interleave, so a `response_mask` over a single sequence is the right object. A VLA rollout is *a sequence of independent forward passes*: each step's input is the current images + state + instruction (no accumulated context), and its output is one chunk of action tokens or continuous actions. The training batch is `(B, num_chunks, tokens_per_chunk)` or `(B, num_chunks, H, D)`, every output token is policy-generated, and the mask is about **episode end inside the chunk grid**, not about environment tokens (`SVR/.../ray_trainer.py` L128–131; RLinf `valid_action_mask`).

### 4.1 Transfers as is (the math is unchanged)
| Exercise | Why it transfers | VLA source using the same thing |
|---|---|---|
| `group_relative_advantage`, `rloo_advantage` | Scalar per-rollout outcome, normalized within a group; only the group key changes (initial state, not prompt) | SVR `core_algos.py` L173; RIPT RLOO |
| `dapo_dynamic_sampling` | Exactly SimpleVLA-RL §3.3 and RIPT's all-success/all-fail filter | SVR §3.3; RIPT L269–271 |
| `dapo_clip_higher_loss`, `ppo_clipped_policy_loss`, `ppo_value_loss` | Same surrogate; inputs become per action token / per action / per chunk | SVR §3.3 (0.28); RL4VLA PPO; pi_RL PPO |
| `rl_eval_loop` | Already Gymnasium-shaped: seeded resets, `info["success"]`, terminated vs truncated | openpi LIBERO eval; ManiSkill eval |
| `pass_at_k` | Applies to success over repeated trials per init state | — |
| `reinforce_discounted_returns`, `reinforce_policy_loss` | Step-level returns and masked PG are embodiment-neutral | — |
| `truncated_importance_sampling`, `rollout_weight_sync_staleness`, `async_agent_rollout`, `fully_async_rollout_buffer` | Systems concerns are identical (rollout engine vs trainer mismatch, staleness, concurrency); robot sims make rollouts slower, which makes these more important | RLinf system paper (2509.15965) — **not read in detail** |

### 4.2 Transfers partly
| Exercise | What carries over | What changes |
|---|---|---|
| `gae_advantage` | The recursion | Uses one `dones` flag; robot RL must bootstrap at truncation from `V(final_obs)` (ManiSkill `final_values`) and choose step-level vs chunk-level time (RLinf `preprocess_embodied_advantages_inputs`). |
| `rl_eval_loop` | Metrics | No action chunks, no settle steps, no fixed init-state list, no `success_once` vs `success_at_end`. |
| `per_token_logprobs` | Discrete-action VLAs (OpenVLA, RT-2, pi0-FAST) really are token log-probs | Vocabulary is the action-bin tail (see `vla_action_bins`); the reduction over the 7 dims of an action and the chunk is a design choice (RLinf `logprob_type`); continuous L1/flow heads have **no** per-token log-prob. |
| `grpo_token_loss`, `loss_aggregation_modes`, `k3_kl_penalty` | Loss algebra | SimpleVLA-RL drops KL; the "sequence" is a trajectory of many forward passes, so `seq-mean-*` means per episode, and episode lengths vary by 10x. |
| `rollout_batch_assembly`, `response_token_mask` | Right-padding + mask idea | No prompt segment inside the response; mask = `arange < finish_step * tokens_per_action`, padded along the chunk axis. |
| `vineppo_mc_value` | MC value from re-rollouts | Requires restoring simulator state (LIBERO `set_init_state`); RECAP uses MC returns but trains a distributional critic. |
| `act_temporal_ensemble`, `rtc_prefix_guidance`, `pi0_flow_matching_chunk` (embodied-ai) | Policy-side chunk handling | Under RL the executed action must be the *sampled* action whose log-prob is stored; ensembling/guidance changes the executed action and breaks the ratio unless accounted for. pi0 Euler sampling is deterministic (ODE) — RL needs the SDE/noise variants. |

### 4.3 Assumptions that break
- **`agent_env_adapter`, `agentic_rollout_loop`, `agent_rollout_truncation`, `rollout_train_boundary`**: they assume a single token stream where environment output is *context tokens* to be masked out and truncation keeps the oldest prefix of one sequence. In a VLA there are no environment tokens in the response, the observation is an image consumed fresh each step, and truncation is a time limit on env steps, not a context window.
- **Per-token log-probs for flow/diffusion heads**: undefined; needs a stochastic sampler (Flow-SDE / Flow-Noise / ReinFlow) or a likelihood-free method (RECAP, DSRL, PA-RL, residual RL).
- **Reward at the last token**: VLA reward may be per env step (ManiSkill dense, RL4VLA 0.1/1.0) and must be mapped onto chunk boundaries.
- **"Prompt" as group key**: for robots the same instruction with a different initial state is a different problem; groups must be keyed by init state (RIPT, RLinf).
- **Episode boundary handling**: LLM rollouts never auto-reset mid-batch; vector envs do, and the NEXT_STEP reset transition is garbage that must be masked.

---

## 5. Proposed path: "Robot RL Environments" (8 exercises)

**Placement.** A new path `robot-rl-env`, group `training`, prerequisites `["rl-posttraining", "embodied-ai"]`. Reason: `rl-posttraining` already has 39 problems and is LLM-shaped (token streams, tools); `embodied-ai` is in `architectures` and covers policy-side numerics. Exercises 6–8 could alternatively be appended to `rl-posttraining` as a "VLA RL" module; duplicates across paths are allowed in `paths.json` (e.g. `cross_entropy`, `lora` appear twice). Oracles would go in a new `torch_judge/harness/embodied_rl/vendored/` following the `torch_judge/harness/rl/vendored/` docstring convention. All exercises: torch/numpy/stdlib, CPU, seeded `torch.Generator`, no simulator — environments are tiny tensors (a 2-D point-mass with an `H×W` occupancy "image").

Teaching order: env contract → action interface → chunk execution → vectorization/autoreset → metrics → value targets → group advantage → flow-policy likelihood.

**R1. `point_reach_env` — the Gymnasium contract for a robot task** (Medium)
- Contract: class `PointReachEnv(max_steps, goal_radius, image_size)` with `reset(seed=None, options=None) -> (obs, info)` and `step(action) -> (obs, reward, terminated, truncated, info)`. `obs = {"image": (1,S,S) float, "state": (2,) float}`; initial position and goal are drawn from a generator re-seeded only when `seed` is given, or taken from `options["init_state"]`; `terminated` exactly when the goal is reached (`info["success"] = True`, reward 1), `truncated` when `elapsed >= max_steps` and not terminated; stepping after done raises.
- Entry baseline: `rl_eval_loop` (consumer side of the same API).
- Oracle: `GYM/gymnasium/core.py` (`Env.reset`/`step` semantics, L100–150), `GYM/gymnasium/utils/env_checker.py::check_reset_seed_determinism` (L79) as a test, `GYM/gymnasium/wrappers/common.py::TimeLimit.step` (L120–136) — MIT. Task structure mirrors `MS/.../pick_cube.py::_initialize_episode/evaluate` (Apache-2.0).
- Mutations: success reported as truncation; reset re-seeding on every unseeded call; `elapsed > max_steps` off by one.

**R2. `vla_action_adapter` — model output to env command** (Easy–Medium)
- Contract: `adapt_actions(actions (…, Dpad), q01 (D,), q99 (D,), state (D,), delta_mask (D,), gripper="binary_flip") -> (…, D)`: quantile-unnormalize the first `D` dims with `(x+1)/2*(q99-q01+1e-6)+q01`, drop padded dims, add `state` on masked dims (delta → absolute), map gripper [0,1] → [-1,1], `sign` binarize, flip sign, clip to bounds.
- Entry baseline: `vla_action_bins` (it already covers q01/q99 on the encode side; this is the decode-to-env side and must not repeat binning).
- Oracle: `OPI/src/openpi/transforms.py::Unnormalize._unnormalize_quantile` (L175–181), `AbsoluteActions` (L226–238), `pad_to_dim` (L423) — Apache-2.0; `SVR/verl/utils/libero_utils.py::normalize_gripper_action/invert_gripper_action` (L156, L196) — MIT.
- Edge case worth testing: gripper exactly 0.5 binarizes to 0 under `np.sign` (upstream behavior; contract must state it).
- Mutations: unnormalize with stats of padded width; delta added on unmasked dims; flip before binarize.

**R3. `chunk_step_executor` — executing an action chunk with mid-chunk termination** (Hard)
- Contract: `chunk_step(env_step, chunk (B,H,D), done_before (B,)) -> dict` where `env_step(actions (B,D)) -> (obs, reward, terminated, truncated)` is batched. Return per-env `valid_mask (B,H)` (prefix, includes the ending action), `rewards (B,H)` zero after the end, `chunk_reward (B,)`, `terminated/truncated (B,)` folded to chunk level, `final_obs` = observation at the ending step (not later), and `num_executed (B,)`. Envs already done never contribute.
- Entry baseline: `act_temporal_ensemble` / `embodied_action_stream` (chunks on the policy side; this is the env side).
- Oracle: `RLINF/rlinf/envs/sim/libero/libero_env.py::chunk_step` (L933–1030) + `RLINF/rlinf/envs/utils.py::valid_action_mask_from_counts` (L134) — Apache-2.0; `MS/mani_skill/utils/wrappers/action_repeat.py` — Apache-2.0. DSRL's wrapper (keeps stepping, no license) is the documented anti-pattern.
- Mutations: mask excludes the ending action; reward summed over the whole chunk; `final_obs` taken from the last chunk slot.

**R4. `vector_autoreset` — NEXT_STEP vs SAME_STEP** (Hard)
- Contract: class `SyncVec(envs, mode)` with `reset(seed)` (env i gets `seed + i`) and `step(actions)`. NEXT_STEP: an env flagged done last call is reset this call, its action ignored, reward 0, flags False, and `info["episode_start"][i] = True`. SAME_STEP: reset in the same call; returned obs is the reset obs, `info["final_obs"][i]`/`info["final_info"][i]` hold the true last ones, with `info["_final_obs"]` masks.
- Entry baseline: `R1`.
- Oracle: `GYM/gymnasium/vector/sync_vector_env.py::SyncVectorEnv.step` (L275–350) and `AutoresetMode` (`vector_env.py` L41–46) — MIT; ManiSkill SAME_STEP naming (`final_observation`) noted as a variant.
- Mutations: reset obs written into `final_obs`; NEXT_STEP reset transition keeps the stale reward; per-env seeds all equal.

**R5. `episode_metrics_tracker` — success_once, success_at_end, partial reset** (Medium)
- Contract: class `EpisodeMetrics(num_envs)` with `update(reward, terminated, truncated, success) -> dict | None` and `reset(env_idx=None)`. Track return, length, `success_once` (OR over time), `success_at_end`; on done emit `{"r","l","success_once","success_at_end"}` only for done envs and clear only those; a mode `count_return_after_success=False` stops accumulating return after first success (RLinf LIBERO).
- Entry baseline: `rl_eval_loop`.
- Oracle: `MS/mani_skill/vector/wrappers/gymnasium.py` (L80–157) — Apache-2.0; `GYM/gymnasium/wrappers/common.py::RecordEpisodeStatistics` (L468+) — MIT; `RLINF/.../libero_env.py::_record_metrics` (L732–756) — Apache-2.0.
- Mutations: clearing all envs on a partial reset; `success_at_end` computed as `success_once`; length counted from 0.

**R6. `truncation_bootstrap_gae` — value targets under autoreset and chunks** (Hard)
- Contract: `robot_gae(rewards (T,B), values (T,B), next_value (B,), terminated (T,B), truncated (T,B), final_values (T,B), gamma, lam) -> (advantages, returns)`: at a terminated step the next value is 0; at a truncated step it is `final_values[t]` (V of the true final obs); otherwise `values[t+1]` (or `next_value` at T−1); the λ-trace is cut at both. Option `chunk_level=True` takes `(T,B,H)` step rewards and a `(T,B,H)` valid mask, sums valid rewards per chunk, and runs GAE over chunks.
- Entry baseline: `gae_advantage` (which has a single `dones`).
- Oracle: `MS/examples/baselines/ppo/ppo.py` L274, L331, L337–348 (`real_next_values = next_not_done * nextvalues + final_values[t]`) — Apache-2.0; chunk reshaping `RLINF/rlinf/algorithms/utils.py::preprocess_embodied_advantages_inputs` (L69–90) — Apache-2.0. Note: the oracle must be the ManiSkill form, since RLinf's GAE does not separate truncation.
- Mutations: bootstrap on termination; zero on truncation; trace not cut at truncation.

**R7. `init_state_group_advantage` — GRPO/RLOO for robot rollouts** (Medium–Hard)
- Contract: `robot_group_advantage(success (N,), group_ids (N,), finish_steps (N,), tokens_per_action, num_slots, mode="grpo"|"rloo", drop_uniform=True) -> dict`: group by init-state id (not by instruction), drop all-success/all-fail groups, compute GRPO (`(r-mean)/(std+eps)`, group of size 1 → mean 0, std 1) or RLOO, broadcast to a `(N, num_slots)` token grid masked by `arange < finish_steps * tokens_per_action`.
- Entry baseline: `group_relative_advantage`, `dapo_dynamic_sampling`, `response_token_mask`.
- Oracle: `SVR/verl/trainer/ppo/core_algos.py::compute_grpo_outcome_advantage` (L173–215, note `torch.std` unbiased) and `SVR/verl/trainer/ppo/ray_trainer.py::compute_advantage` mask (L124–131) — MIT. RIPT's init-state grouping is cited only (no license).
- Mutations: grouping by instruction string; mask uses chunk count instead of env steps; dropped groups still contribute.

**R8. `flow_sde_chain_logprob` — a likelihood for a flow-matching action head** (Hard, capstone)
- Contract: `flow_step_mean_std(x_t, v_t, idx, num_steps, method, noise_level) -> (mean, std)` for `flow_ode` (std 0) and `flow_sde` (`sigma_i = noise_level*sqrt(t/(1-t))`, with the `t=1` denominator replaced by the next timestep), using the noise-at-one convention; and `chain_logprob(xs (K+1,B,H,D), means, stds, level) -> Tensor` summing per-step Gaussian log-densities (zero where std = 0) and reducing to `token_level` / `action_level` / `chunk_level`.
- Entry baseline: `pi0_flow_matching_chunk` (ODE Euler sampler, same convention) and `ppo_clipped_policy_loss`.
- Oracle: `RLINF/rlinf/models/embodiment/openpi/sampling/rl_sampler.py::get_timesteps/sample_mean_var/gaussian_logprob` (L24–102) and `RLINF/rlinf/algorithms/utils.py::preprocess_loss_inputs` (L282–380) — Apache-2.0. Paper: pi_RL §4.1–4.2, ReinFlow Eq. 6–7.
- Mutations: SDE correction term `sigma²·δ/(2t)` dropped (mean then matches ODE); log-prob computed for the final action only instead of the chain; `chunk_level` sums over the batch axis.

**Not recommended for now**
| Candidate | Why not |
|---|---|
| `domain_randomization_sampler` (seeded per-episode physics/visual params, startup/reset/interval) | Small; Isaac Lab event modes not verified from source; fits as a visible example inside R1. |
| `recap_advantage_indicator` (binarize advantages at a per-task percentile, drop the indicator for CFG) | Good, but pi\*0.6 has no code; the contract would rest on the paper alone. Candidate for a later "likelihood-free" module with DSRL and residual RL. |
| `dsrl_latent_env` (wrap a frozen policy so the action is its noise) | No upstream license; would be an original contract. |
| `residual_policy_mix` (base + α·tanh residual, ε ramp) | Policy Decorator code not pinned; small. |

---

## 6. Open questions / unverified
- RLinf system paper (2509.15965) and RLinf-VLA paper body were not read; only the code at the pinned commit.
- Isaac Lab `EventTerm` mode names and `time_out` semantics were taken from docs/API summaries, not source.
- LIBERO per-suite contents (Spatial/Object/Goal/Long, 10 tasks each, 50 demos per task) are **Unverified** here; only "4 suites, 130 tasks" was read.
- ManiSkill's PPO comment references a `bootstrap_at_done` option that no longer exists as an argument at this commit (`ppo.py` L348).
