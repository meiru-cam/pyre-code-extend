# Advanced AI Systems Curriculum Resources

This is the curated primary-source index for authoring the Advanced Attention, MoE, Agent Runtime, and Agent Guardrails paths. Individual exercises must link to precise sections and commit-pinned code locations rather than only to these top-level resources.

## Knowledge

### Frontier attention and model architecture

- [Kimi K2 technical report](https://arxiv.org/abs/2507.20534) and [official Kimi K2 repository](https://github.com/moonshotai/Kimi-K2)
  Use for: MLA, large-scale MoE configuration, routing, training stability, and deployment-oriented implementation references.
- [GLM-4.5 technical report](https://arxiv.org/abs/2508.06471) and [official GLM-4.5 repository](https://github.com/zai-org/GLM-4.5)
  Use for: hybrid model architecture and multi-stage training connections.
- [Qwen3 technical report](https://arxiv.org/abs/2505.09388) and [official Qwen3 repository](https://github.com/QwenLM/Qwen3)
  Use for: Qwen attention and MoE architecture, routing behavior, and model-family comparisons.
- [Meta's official Llama 4 architecture announcement](https://ai.meta.com/blog/llama-4-multimodal-intelligence/)
  Use for: iRoPE and shared/routed expert architecture context.
- [Gemma 3 technical report](https://arxiv.org/abs/2503.19786)
  Use for: local/global attention schedules and their efficiency tradeoffs.
- [Hugging Face Transformers Mistral implementation](https://github.com/huggingface/transformers/tree/main/src/transformers/models/mistral)
  Use for: sliding-window attention code mapping. Pin a Transformers commit in every exercise.

### MoE routing and training

- [Switch Transformers](https://www.jmlr.org/papers/v23/21-0998.html)
  Use for: sparse routing, capacity factors, token dropping, router losses, and training-stability foundations.
- [DeepSeek-V3 technical report](https://arxiv.org/abs/2412.19437) and [official DeepSeek-V3 repository](https://github.com/deepseek-ai/DeepSeek-V3)
  Use for: fine-grained experts, shared experts, auxiliary-loss-free load balancing, and MoE training connections.
- [Hugging Face Transformers Qwen3-MoE implementation](https://github.com/huggingface/transformers/tree/main/src/transformers/models/qwen3_moe)
  Use for: a readable production implementation of router and expert execution behavior. Pin a commit and exact class or method per exercise.
- [Hugging Face Transformers Mixtral implementation](https://github.com/huggingface/transformers/tree/main/src/transformers/models/mixtral)
  Use for: sparse top-k routing and expert execution comparisons. Pin a commit per exercise.

### Agent runtime and multi-agent systems

- [OpenClaw official repository](https://github.com/openclaw/openclaw)
  Use for: agent-loop events, tool execution, typed gateway frames, retry/failover control, queue draining, and delegation patterns.
- [Hermes Agent official repository](https://github.com/NousResearch/hermes-agent)
  Use for: iteration budgets, retry classification, tool concurrency, tool guardrails, delegation, and session delivery.
- [Microsoft Agent Framework official repository](https://github.com/microsoft/agent-framework)
  Use for: workflow graphs, fan-out/fan-in, superstep execution, concurrency, checkpointing, and restore behavior.

### Guardrails and security

- [NVIDIA NeMo Guardrails official repository](https://github.com/NVIDIA/NeMo-Guardrails)
  Use for: input, output, dialog, retrieval, and execution rail concepts.
- [Meta Purple Llama official repository](https://github.com/meta-llama/PurpleLlama)
  Use for: prompt-guard and security-evaluation patterns.
- [OpenAI Agents SDK guardrails documentation](https://openai.github.io/openai-agents-python/guardrails/)
  Use for: input, output, and tool guardrail lifecycle and failure behavior.
- [OWASP Agentic Security Initiative](https://genai.owasp.org/initiatives/agentic-security-initiative/)
  Use for: agent-specific threat modeling, least privilege, memory and tool risks, and defensive terminology.

### Optional AI feedback

- [DeepSeek API documentation](https://api-docs.deepseek.com/)
  Use for: the OpenAI-compatible provider adapter and current model configuration.
- [DeepSeek models and pricing](https://api-docs.deepseek.com/quick_start/pricing)
  Use for: selecting the recommended current reviewer model without pinning the curriculum to deprecated aliases.

## Wisdom (Communities)

- The issue trackers and discussions attached to the official repositories above
  Use for: implementation edge cases, maintainer explanations, and production failures that are not fully described in papers. Treat comments as practical context, not normative specifications.

## Gaps

- A stable, primary source that specifies a common cross-framework agent message protocol; exercises should therefore teach an explicit minimal protocol and compare multiple implementations.
- Long-term empirical studies of AI-based system-design review consistency; AI feedback remains optional and non-authoritative.
