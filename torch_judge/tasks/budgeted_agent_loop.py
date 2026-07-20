"""A deterministic tool-using agent loop with explicit operational limits."""

from torch_judge.tasks._schema import build_design_note_rubric

TASK = {
    "title": "Budgeted Retrying Agent Loop",
    "difficulty": "Hard",
    "version": 1,
    "function_name": "budgeted_agent_loop",
    "description_en": r"""Implement a deterministic tool-using agent state machine.

Signature: 'budgeted_agent_loop(model, registry, messages, limits, retry_policy, clock, trace) -> AgentResult'. Import 'AgentResult', 'RetryableModelError', 'PermanentModelError', 'RetryableToolError', and 'PermanentToolError' from 'torch_judge.harness.agents'. Do not sleep or call a live API.

'model.call(messages)' returns one JSON-like response per iteration:

- final: '{"type":"final", "content":str, "usage":{"tokens":int,"cost":number}}'
- tool call: '{"type":"tool_call", "call_id":str, "name":str, "arguments":dict, "idempotency_key":str|None, "usage":...}'

'limits' contains non-negative 'max_iterations', 'max_tool_calls', 'max_tokens', 'max_time', and 'max_cost'. It may contain 'cancelled', either a boolean or zero-argument callable. Every model attempt—including a provider failure—consumes one iteration; a successful response consumes its usage immediately. A tool attempt—including a failed attempt—consumes one tool call. Values equal to a maximum are allowed; the next unit is not. 'retry_policy' contains positive 'max_attempts' (the first call counts) and non-negative 'base_delay'. Retry 'RetryableModelError', 'TimeoutError', and 'RetryableToolError'; prefer an available 'error.retry_after', otherwise use exponential delays 'base_delay * 2 ** (attempt - 1)'. Permanent model/tool failures and unexpected adapter errors terminate with distinct typed failures. Reuse one tool idempotency key across attempts, deriving it from 'call_id' when absent. A 'HarnessFailure' denotes a broken evaluator fixture and must escape unchanged.

Return status 'completed', 'failed', 'budget_exceeded', or 'cancelled'. Failed/budget results carry a JSON-like 'failure' with a stable 'type'; never turn a permanent failure, retry exhaustion, malformed response/result, or budget violation into success. Append successful tool results to an isolated message list. Emit correlated trace events using run/correlation id 'agent-run': 'run.started'; 'model.called' or 'model.failed' plus 'model.retry'; 'tool.started', 'tool.retry', 'tool.completed' or 'tool.failed'; then a terminal 'run.completed', 'run.failed', 'run.budget_exceeded', or 'run.cancelled'.

This is a compact synchronous harness. Production systems additionally stream tokens, abort in-flight I/O, classify provider-specific errors, persist state, and coordinate concurrent calls.

Optional code reading: OpenClaw's 'runEmbeddedAttempt' wires aborts, tool-result context guards, and run lifecycle around a provider attempt. Hermes 'AIAgent.run_conversation' owns its main tool loop, while gateway API runs expose bounded concurrency and structured tool events.""",
    "advisory_prerequisites": ["tool_registry"],
    "design_note_rubric": build_design_note_rubric(),
    "hints": [
        {"level": 1, "kind": "questions", "content": "At which exact boundary is each counter consumed? Does a failed attempt count? Which key stays stable across retries? Can a retry delay itself exceed the time budget? What terminal trace should every return path emit?"},
        {"level": 2, "kind": "analysis", "content": "Copy messages and record a start time. Before each model and tool attempt check cancellation and the relevant next-unit budget; count failed model calls as iterations, then add usage only after a valid response. Retry RetryableModelError, TimeoutError, and RetryableToolError; choose retry_after before exponential backoff and reject a delay that would cross max_time. Let HarnessFailure escape, classify permanent and unexpected runtime errors distinctly, and route every learner-facing exit through one terminal-event helper."},
    ],
    "model_connections": [
        "OpenClaw runEmbeddedAttempt combines an AbortController, model/provider context, tool-result guards, hooks, and run telemetry; this exercise makes a smaller state transition table explicit.",
        "Hermes AIAgent.run_conversation contains the main conversational tool loop, and its API server emits tool.started/tool.completed events while bounding concurrent runs.",
        "The stable internal result and error taxonomy is an adapter boundary: provider-specific rate limits and timeouts can be classified without leaking provider formats into loop policy.",
    ],
    "pro_con_analysis": {
        "pros": [
            "Explicit independent budgets bound runaway loops, spend, latency, and side effects.",
            "Virtual time makes retry and timeout behavior fast and exactly reproducible.",
            "Stable idempotency keys make retry safety testable instead of aspirational.",
        ],
        "cons": [
            "A synchronous loop cannot overlap independent tools or cancel real in-flight I/O.",
            "Fail-fast permanent errors are simple but give the model no opportunity to recover with another plan.",
            "Locally counted tokens and cost rely on trustworthy provider usage adapters.",
        ],
    },
    "sources": [
        {"kind": "code", "url": "https://github.com/openclaw/openclaw", "commit": "40d31f34813c2a01284b097c0d0d785fbb173400", "path": "src/agents/pi-embedded-runner/run/attempt.ts", "symbol": "runEmbeddedAttempt", "license": "MIT", "adapted": "Run lifecycle, abort ownership, provider attempt boundary, tool-result guarding, and operational telemetry.", "simplifications": "Synchronous scripted model, no streaming, provider fallback, hooks, context compaction, sandbox, or real abort signal."},
        {"kind": "code", "url": "https://github.com/NousResearch/hermes-agent", "commit": "299e409f15aa5615a8a64be488580be92cda351e", "path": "run_agent.py", "symbol": "AIAgent.run_conversation", "license": "MIT", "adapted": "Conversation-owned model/tool iteration and terminal response flow.", "simplifications": "No memory, skills, compression, provider routing, streaming, persistent session, or interactive interruption."},
        {"kind": "code", "url": "https://github.com/NousResearch/hermes-agent", "commit": "299e409f15aa5615a8a64be488580be92cda351e", "path": "gateway/platforms/api_server.py", "symbol": "ApiServerPlatform._handle_runs and _make_run_event_callback", "license": "MIT", "adapted": "Bounded run allocation and structured tool lifecycle events.", "simplifications": "One local run with an in-memory trace instead of HTTP, SSE queues, threads, authentication, and TTL cleanup."},
    ],
    "tests": [
        {"name": "Retry-after, idempotency, messages, and trace", "behavior": "retry.backoff", "code": r"""
from torch_judge.harness.agents import FakeTool,RetryableToolError,ScriptedModel,TraceRecorder,VirtualClock
class Registry:
    def __init__(self,tool): self.tool=tool
    def invoke(self,name,arguments,idempotency_key=None):
        if name!=self.tool.name: raise KeyError(name)
        return self.tool.invoke(arguments,idempotency_key)
tool=FakeTool('search','search',{'q':str},outcomes=[RetryableToolError('rate',retry_after=1.5),{'answer':'ok'}])
model=ScriptedModel([
 {'type':'tool_call','call_id':'call-1','name':'search','arguments':{'q':'x'},'idempotency_key':None,'usage':{'tokens':4,'cost':.1}},
 {'type':'final','content':'done','usage':{'tokens':2,'cost':.05}},
])
clock=VirtualClock(); trace=TraceRecorder(); registry=Registry(tool)
limits={'max_iterations':3,'max_tool_calls':3,'max_tokens':10,'max_time':5,'max_cost':1}
result={fn}(model,registry,[{'role':'user','content':'go'}],limits,{'max_attempts':2,'base_delay':.25},clock,trace)
assert result.status=='completed' and result.content=='done'
assert (result.iterations,result.tool_calls,result.tokens,result.elapsed)==(2,2,6,1.5)
assert abs(result.cost-.15)<1e-12
assert tool.calls==2 and tool.effects==1 and tool.effect_keys==['call-1']
assert tool.invocation_keys==['call-1','call-1']
assert clock.sleeps==[1.5]
assert model.calls[1][-1]=={'role':'tool','tool_call_id':'call-1','name':'search','content':{'answer':'ok'}}
assert [event.kind for event in trace.events]==['run.started','model.called','tool.started','tool.retry','tool.started','tool.completed','model.called','run.completed']
"""},
        {"name": "Permanent failures are terminal and typed", "behavior": "retry.classification", "code": r"""
from torch_judge.harness.agents import FakeTool,PermanentToolError,ScriptedModel,TraceRecorder,VirtualClock
class Registry:
    def __init__(self,tool): self.tool=tool
    def invoke(self,name,arguments,idempotency_key=None): return self.tool.invoke(arguments,idempotency_key)
tool=FakeTool('delete','delete',{},outcomes=[PermanentToolError('denied')])
model=ScriptedModel([{'type':'tool_call','call_id':'c','name':'delete','arguments':{},'idempotency_key':'k','usage':{'tokens':1,'cost':0}}])
trace=TraceRecorder()
result={fn}(model,Registry(tool),[],{'max_iterations':2,'max_tool_calls':3,'max_tokens':3,'max_time':2,'max_cost':1},{'max_attempts':3,'base_delay':1},VirtualClock(),trace)
assert result.status=='failed' and result.failure=={'type':'permanent_tool_failure','tool':'delete','message':'denied'}
assert result.tool_calls==1 and tool.calls==1
assert [e.kind for e in trace.events][-2:]==['tool.failed','run.failed']
"""},
        {"name": "Every cumulative budget has an exact boundary", "behavior": "budget.enforcement", "visibility": "unshown", "failure_message": "Allow work equal to a limit, count every attempt, and stop before the next unit or retry delay crosses any budget.", "code": r"""
from torch_judge.harness.agents import FakeTool,RetryableToolError,ScriptedModel,TraceRecorder,VirtualClock
class Registry:
    def __init__(self,tool): self.tool=tool
    def invoke(self,name,arguments,idempotency_key=None): return self.tool.invoke(arguments,idempotency_key)
def run(model,tool,limits,policy={'max_attempts':2,'base_delay':2}):
    return {fn}(model,Registry(tool),[],limits,policy,VirtualClock(),TraceRecorder())
base={'max_iterations':1,'max_tool_calls':1,'max_tokens':2,'max_time':2,'max_cost':.5}
# Equality is valid.
r=run(ScriptedModel([{'type':'final','content':'ok','usage':{'tokens':2,'cost':.5}}]),FakeTool('x','x',{},outcomes=[]),base)
assert r.status=='completed'
# Iteration, tokens, and cost each fail distinctly.
for changed,expected in [({'max_iterations':0},'iterations'),({'max_tokens':1},'tokens'),({'max_cost':.49},'cost')]:
    limits={**base,**changed}; model=ScriptedModel([{'type':'final','content':'x','usage':{'tokens':2,'cost':.5}}])
    r=run(model,FakeTool('x','x',{},outcomes=[]),limits)
    assert r.status=='budget_exceeded' and r.failure['limit']==expected
# Failed attempts consume calls; retry delay cannot cross time.
tool=FakeTool('x','x',{},outcomes=[RetryableToolError('busy'),{}])
script=[{'type':'tool_call','call_id':'c','name':'x','arguments':{},'usage':{'tokens':0,'cost':0}},{'type':'final','content':'done','usage':{'tokens':0,'cost':0}}]
model=ScriptedModel(script)
r=run(model,tool,{**base,'max_time':1,'max_iterations':2})
assert r.status=='budget_exceeded' and r.failure['limit']=='time' and tool.calls==1
tool=FakeTool('x','x',{},outcomes=[RetryableToolError('busy'),{}])
r=run(ScriptedModel(script),tool,{**base,'max_time':5,'max_iterations':2})
assert r.status=='budget_exceeded' and r.failure['limit']=='tool_calls' and tool.calls==1
"""},
        {"name": "Cancellation and malformed responses fail closed", "behavior": "state.invariant", "visibility": "unshown", "failure_message": "Check cancellation at work boundaries and return typed failure for malformed model or tool data.", "code": r"""
from torch_judge.harness.agents import ScriptedModel,TraceRecorder,VirtualClock
class NeverRegistry:
    def invoke(self,*args,**kwargs): raise AssertionError('must not invoke')
base={'max_iterations':2,'max_tool_calls':2,'max_tokens':5,'max_time':5,'max_cost':1}
r={fn}(ScriptedModel([{'type':'final','content':'no','usage':{'tokens':0,'cost':0}}]),NeverRegistry(),[],{**base,'cancelled':True},{'max_attempts':1,'base_delay':0},VirtualClock(),TraceRecorder())
assert r.status=='cancelled' and r.iterations==0
bad=[
 {'type':'unknown','usage':{'tokens':0,'cost':0}},
 {'type':'final','content':3,'usage':{'tokens':0,'cost':0}},
 {'type':'tool_call','call_id':'','name':'x','arguments':{},'usage':{'tokens':0,'cost':0}},
 {'type':'final','content':'x','usage':{'tokens':-1,'cost':0}},
]
for response in bad:
    r={fn}(ScriptedModel([response]),NeverRegistry(),[],base,{'max_attempts':1,'base_delay':0},VirtualClock(),TraceRecorder())
    assert r.status=='failed' and r.failure['type']=='malformed_model_response'
class BadRegistry:
    def invoke(self,*args,**kwargs): return object()
response={'type':'tool_call','call_id':'c','name':'x','arguments':{},'usage':{'tokens':0,'cost':0}}
r={fn}(ScriptedModel([response]),BadRegistry(),[],base,{'max_attempts':1,'base_delay':0},VirtualClock(),TraceRecorder())
assert r.status=='failed' and r.failure['type']=='malformed_tool_result'
"""},
        {"name": "Retry exhaustion is not successful", "behavior": "retry.classification", "visibility": "unshown", "failure_message": "Retry only retryable failures up to max_attempts, then return a typed terminal failure.", "code": r"""
from torch_judge.harness.agents import FakeTool,RetryableToolError,ScriptedModel,TraceRecorder,VirtualClock
class Registry:
    def __init__(self,tool): self.tool=tool
    def invoke(self,name,arguments,idempotency_key=None): return self.tool.invoke(arguments,idempotency_key)
tool=FakeTool('api','api',{},outcomes=[RetryableToolError('busy'),RetryableToolError('still busy')])
model=ScriptedModel([{'type':'tool_call','call_id':'c','name':'api','arguments':{},'usage':{'tokens':0,'cost':0}}])
clock=VirtualClock(); trace=TraceRecorder()
r={fn}(model,Registry(tool),[],{'max_iterations':1,'max_tool_calls':2,'max_tokens':1,'max_time':5,'max_cost':1},{'max_attempts':2,'base_delay':1},clock,trace)
assert r.status=='failed' and r.failure=={'type':'retry_exhausted','tool':'api','attempts':2,'message':'still busy'}
assert tool.calls==2 and clock.sleeps==[1.0]
"""},
        {"name": "Registry and handler errors become typed failures", "behavior": "retry.classification", "visibility": "unshown", "failure_message": "Unexpected registry or handler errors must terminate with a typed tool-execution failure rather than escape or look successful.", "code": r"""
from torch_judge.harness.agents import ScriptedModel,TraceRecorder,VirtualClock
class BrokenRegistry:
    def __init__(self,error): self.error=error
    def invoke(self,*args,**kwargs): raise self.error
response={'type':'tool_call','call_id':'c','name':'missing','arguments':{},'usage':{'tokens':0,'cost':0}}
limits={'max_iterations':1,'max_tool_calls':1,'max_tokens':1,'max_time':1,'max_cost':1}
for error in (KeyError('missing'),RuntimeError('adapter broke')):
    r={fn}(ScriptedModel([response]),BrokenRegistry(error),[],limits,{'max_attempts':2,'base_delay':0},VirtualClock(),TraceRecorder())
    assert r.status=='failed'
    assert r.failure=={'type':'tool_execution_failure','tool':'missing','error_type':type(error).__name__,'message':str(error)}
    assert r.tool_calls==1
"""},
        {"name": "Model API failures retry or terminate with typed results", "behavior": "retry.classification", "visibility": "unshown", "failure_message": "Provider failures must consume iterations, retry only retryable failures with virtual time, and always emit a typed terminal result.", "code": r"""
from torch_judge.harness.agents import PermanentModelError,RetryableModelError,ScriptedModel,TraceRecorder,VirtualClock
class NeverRegistry:
    def invoke(self,*args,**kwargs): raise AssertionError('must not invoke')
limits={'max_iterations':3,'max_tool_calls':0,'max_tokens':1,'max_time':5,'max_cost':1}
clock=VirtualClock(); trace=TraceRecorder()
r={fn}(ScriptedModel([RetryableModelError('rate',retry_after=1.25),{'type':'final','content':'ok','usage':{'tokens':0,'cost':0}}]),NeverRegistry(),[],limits,{'max_attempts':2,'base_delay':.5},clock,trace)
assert r.status=='completed' and r.iterations==2 and clock.sleeps==[1.25]
assert [e.kind for e in trace.events]==['run.started','model.failed','model.retry','model.called','run.completed']
r={fn}(ScriptedModel([PermanentModelError('bad auth')]),NeverRegistry(),[],limits,{'max_attempts':2,'base_delay':0},VirtualClock(),TraceRecorder())
assert r.status=='failed' and r.iterations==1
assert r.failure=={'type':'permanent_model_failure','message':'bad auth'}
r={fn}(ScriptedModel([RuntimeError('adapter broke')]),NeverRegistry(),[],limits,{'max_attempts':2,'base_delay':0},VirtualClock(),TraceRecorder())
assert r.status=='failed' and r.failure=={'type':'model_execution_failure','error_type':'RuntimeError','message':'adapter broke'}
"""},
        {"name": "Harness author failures remain evaluator errors", "behavior": "state.invariant", "visibility": "unshown", "failure_message": "An exhausted scripted model or tool is an invalid evaluator fixture and must not be converted into a learner result.", "code": r"""
from torch_judge.harness import HarnessFailure
from torch_judge.harness.agents import FakeTool,ScriptedModel,TraceRecorder,VirtualClock
class Registry:
    def __init__(self,tool): self.tool=tool
    def invoke(self,*args,**kwargs): return self.tool.invoke({},'k')
limits={'max_iterations':1,'max_tool_calls':1,'max_tokens':1,'max_time':1,'max_cost':1}
policy={'max_attempts':1,'base_delay':0}
try:
    {fn}(ScriptedModel([]),Registry(FakeTool('x','x',{},outcomes=[])),[],limits,policy,VirtualClock(),TraceRecorder())
except HarnessFailure: pass
else: raise AssertionError('model exhaustion must escape')
response={'type':'tool_call','call_id':'c','name':'x','arguments':{},'usage':{'tokens':0,'cost':0}}
try:
    {fn}(ScriptedModel([response]),Registry(FakeTool('x','x',{},outcomes=[])),[],limits,policy,VirtualClock(),TraceRecorder())
except HarnessFailure: pass
else: raise AssertionError('tool exhaustion must escape')
"""},
    ],
    "solution": r'''def budgeted_agent_loop(model, registry, messages, limits, retry_policy, clock, trace):
    from copy import deepcopy
    from math import isfinite
    from torch_judge.harness import HarnessFailure
    from torch_judge.harness.agents import (AgentResult, PermanentModelError,
        PermanentToolError, RetryableModelError, RetryableToolError)
    from torch_judge.harness.agents.protocol import copy_json

    run_id = correlation_id = "agent-run"
    start = clock.now()
    state = deepcopy(messages)
    iterations = tool_calls = tokens = 0
    cost = 0.0

    required = ("max_iterations", "max_tool_calls", "max_tokens", "max_time", "max_cost")
    if any(name not in limits for name in required):
        raise ValueError("limits are incomplete")
    if any(isinstance(limits[name], bool) or not isinstance(limits[name], (int, float))
           or not isfinite(float(limits[name])) or limits[name] < 0 for name in required):
        raise ValueError("limits must be non-negative finite numbers")
    max_attempts = retry_policy.get("max_attempts")
    base_delay = retry_policy.get("base_delay")
    if not isinstance(max_attempts, int) or isinstance(max_attempts, bool) or max_attempts < 1:
        raise ValueError("max_attempts must be a positive integer")
    if isinstance(base_delay, bool) or not isinstance(base_delay, (int, float)) or base_delay < 0:
        raise ValueError("base_delay must be non-negative")

    def elapsed():
        return clock.now() - start

    def cancelled():
        value = limits.get("cancelled", False)
        return bool(value() if callable(value) else value)

    terminal_events = {
        "completed": "run.completed", "failed": "run.failed",
        "budget_exceeded": "run.budget_exceeded", "cancelled": "run.cancelled",
    }

    def finish(status, content=None, failure=None):
        trace.emit(terminal_events[status], clock.now(), run_id, correlation_id,
                   {} if failure is None else {"failure": failure})
        return AgentResult(status, content, state, iterations, tool_calls, tokens,
                           cost, elapsed(), failure)

    def budget_failure(limit):
        return finish("budget_exceeded", failure={"type": "budget", "limit": limit})

    trace.emit("run.started", clock.now(), run_id, correlation_id, {})
    while True:
        if cancelled():
            return finish("cancelled", failure={"type": "cancelled"})
        if elapsed() > limits["max_time"]:
            return budget_failure("time")
        if iterations >= limits["max_iterations"]:
            return budget_failure("iterations")

        for model_attempt in range(1, max_attempts + 1):
            if cancelled():
                return finish("cancelled", failure={"type": "cancelled"})
            if elapsed() > limits["max_time"]:
                return budget_failure("time")
            if iterations >= limits["max_iterations"]:
                return budget_failure("iterations")
            iterations += 1
            try:
                response = model.call(deepcopy(state))
                trace.emit("model.called", clock.now(), run_id, correlation_id,
                           {"iteration": iterations, "attempt": model_attempt})
                break
            except HarnessFailure:
                raise
            except PermanentModelError as error:
                trace.emit("model.failed", clock.now(), run_id, correlation_id,
                           {"attempt": model_attempt, "type": "permanent"})
                return finish("failed", failure={"type": "permanent_model_failure",
                                                  "message": str(error)})
            except (RetryableModelError, TimeoutError) as error:
                trace.emit("model.failed", clock.now(), run_id, correlation_id,
                           {"attempt": model_attempt, "type": "retryable"})
                if model_attempt == max_attempts:
                    return finish("failed", failure={"type": "model_retry_exhausted",
                                                      "attempts": model_attempt,
                                                      "message": str(error)})
                delay = (error.retry_after if isinstance(error, RetryableModelError)
                         and error.retry_after is not None
                         else base_delay * 2 ** (model_attempt - 1))
                if elapsed() + delay > limits["max_time"]:
                    return budget_failure("time")
                trace.emit("model.retry", clock.now(), run_id, correlation_id,
                           {"attempt": model_attempt, "delay": delay})
                clock.sleep(delay)
            except Exception as error:
                trace.emit("model.failed", clock.now(), run_id, correlation_id,
                           {"attempt": model_attempt, "type": "unexpected",
                            "error_type": type(error).__name__})
                return finish("failed", failure={"type": "model_execution_failure",
                                                  "error_type": type(error).__name__,
                                                  "message": str(error)})
        try:
            if not isinstance(response, dict):
                raise ValueError
            usage = response.get("usage")
            used_tokens = usage.get("tokens") if isinstance(usage, dict) else None
            used_cost = usage.get("cost") if isinstance(usage, dict) else None
            if (not isinstance(used_tokens, int) or isinstance(used_tokens, bool)
                    or used_tokens < 0):
                raise ValueError
            if (not isinstance(used_cost, (int, float)) or isinstance(used_cost, bool)
                    or not isfinite(float(used_cost)) or used_cost < 0):
                raise ValueError
        except (AttributeError, ValueError):
            return finish("failed", failure={"type": "malformed_model_response"})
        tokens += used_tokens
        cost += float(used_cost)
        if tokens > limits["max_tokens"]:
            return budget_failure("tokens")
        if cost > limits["max_cost"]:
            return budget_failure("cost")
        if elapsed() > limits["max_time"]:
            return budget_failure("time")

        response_type = response.get("type")
        if response_type == "final":
            if not isinstance(response.get("content"), str):
                return finish("failed", failure={"type": "malformed_model_response"})
            return finish("completed", content=response["content"])
        if response_type != "tool_call":
            return finish("failed", failure={"type": "malformed_model_response"})
        call_id = response.get("call_id")
        name = response.get("name")
        arguments = response.get("arguments")
        idempotency_key = response.get("idempotency_key", None)
        if (not isinstance(call_id, str) or not call_id
                or not isinstance(name, str) or not name
                or not isinstance(arguments, dict)
                or (idempotency_key is not None
                    and (not isinstance(idempotency_key, str) or not idempotency_key))):
            return finish("failed", failure={"type": "malformed_model_response"})
        idempotency_key = idempotency_key or call_id

        for attempt in range(1, max_attempts + 1):
            if cancelled():
                return finish("cancelled", failure={"type": "cancelled"})
            if tool_calls >= limits["max_tool_calls"]:
                return budget_failure("tool_calls")
            if elapsed() > limits["max_time"]:
                return budget_failure("time")
            tool_calls += 1
            trace.emit("tool.started", clock.now(), run_id, correlation_id,
                       {"tool": name, "call_id": call_id, "attempt": attempt})
            try:
                tool_result = registry.invoke(name, deepcopy(arguments), idempotency_key)
                try:
                    tool_result = copy_json(tool_result, field="tool result JSON")
                except Exception:
                    trace.emit("tool.failed", clock.now(), run_id, correlation_id,
                               {"tool": name, "type": "malformed_tool_result"})
                    return finish("failed", failure={"type": "malformed_tool_result", "tool": name})
                state.append({"role": "tool", "tool_call_id": call_id,
                              "name": name, "content": tool_result})
                trace.emit("tool.completed", clock.now(), run_id, correlation_id,
                           {"tool": name, "call_id": call_id, "attempt": attempt})
                break
            except HarnessFailure:
                raise
            except PermanentToolError as error:
                trace.emit("tool.failed", clock.now(), run_id, correlation_id,
                           {"tool": name, "type": "permanent", "attempt": attempt})
                return finish("failed", failure={"type": "permanent_tool_failure",
                                                  "tool": name, "message": str(error)})
            except RetryableToolError as error:
                if attempt == max_attempts:
                    trace.emit("tool.failed", clock.now(), run_id, correlation_id,
                               {"tool": name, "type": "retry_exhausted", "attempt": attempt})
                    return finish("failed", failure={"type": "retry_exhausted", "tool": name,
                                                      "attempts": attempt, "message": str(error)})
                delay = error.retry_after if error.retry_after is not None else base_delay * 2 ** (attempt - 1)
                if elapsed() + delay > limits["max_time"]:
                    return budget_failure("time")
                trace.emit("tool.retry", clock.now(), run_id, correlation_id,
                           {"tool": name, "call_id": call_id, "attempt": attempt, "delay": delay})
                clock.sleep(delay)
            except Exception as error:
                trace.emit("tool.failed", clock.now(), run_id, correlation_id,
                           {"tool": name, "type": "unexpected", "attempt": attempt,
                            "error_type": type(error).__name__})
                return finish("failed", failure={"type": "tool_execution_failure",
                                                  "tool": name,
                                                  "error_type": type(error).__name__,
                                                  "message": str(error)})''',
    "demo": r"""from torch_judge.harness.agents import FakeTool,ScriptedModel,TraceRecorder,VirtualClock
class Registry:
    def __init__(self,tool): self.tool=tool
    def invoke(self,name,arguments,idempotency_key=None): return self.tool.invoke(arguments,idempotency_key)
tool=FakeTool('lookup','lookup',{'id':str},outcomes=[{'value':'Ada'}])
model=ScriptedModel([
 {'type':'tool_call','call_id':'1','name':'lookup','arguments':{'id':'42'},'usage':{'tokens':3,'cost':0}},
 {'type':'final','content':'Ada','usage':{'tokens':2,'cost':0}},
])
result=budgeted_agent_loop(model,Registry(tool),[],{'max_iterations':3,'max_tool_calls':2,'max_tokens':10,'max_time':5,'max_cost':1},{'max_attempts':2,'base_delay':1},VirtualClock(),TraceRecorder())
print(result.status,result.content)""",
}
