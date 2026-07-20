"""Reusable deterministic harness pieces for agent runtime exercises."""

from torch_judge.harness.agents.assertions import (
    assert_budget_usage,
    assert_correlation_chain,
    assert_event_order,
    assert_final_state,
    assert_max_concurrency,
    assert_no_duplicate_effects,
    assert_retry_delays,
)
from torch_judge.harness.agents.fake_tools import FakeTool, PermanentToolError, RetryableToolError
from torch_judge.harness.agents.orchestration import (
    InMemoryBus,
    InMemoryCheckpointStore,
    RoundRobinScheduler,
    ScriptedWorker,
)
from torch_judge.harness.agents.protocol import AgentResult, MessageEnvelope, OrchestrationResult
from torch_judge.harness.agents.scenarios import Scenario
from torch_judge.harness.agents.scripted_model import (
    PermanentModelError,
    RetryableModelError,
    ScriptedModel,
)
from torch_judge.harness.agents.trace import TraceEvent, TraceRecorder
from torch_judge.harness.agents.virtual_clock import VirtualClock

__all__ = [
    "FakeTool",
    "AgentResult",
    "InMemoryBus",
    "InMemoryCheckpointStore",
    "MessageEnvelope",
    "PermanentToolError",
    "PermanentModelError",
    "OrchestrationResult",
    "RoundRobinScheduler",
    "RetryableToolError",
    "RetryableModelError",
    "Scenario",
    "ScriptedModel",
    "ScriptedWorker",
    "TraceEvent",
    "TraceRecorder",
    "VirtualClock",
    "assert_correlation_chain",
    "assert_budget_usage",
    "assert_event_order",
    "assert_final_state",
    "assert_max_concurrency",
    "assert_no_duplicate_effects",
    "assert_retry_delays",
]
