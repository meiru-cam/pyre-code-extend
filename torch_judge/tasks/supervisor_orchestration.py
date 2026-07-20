"""Deterministic supervisor/worker fan-out, retry, and checkpoint recovery."""

from torch_judge.tasks._schema import build_design_note_rubric

TASK = {
    "title": "Resumable Supervisor-Worker Orchestration",
    "difficulty": "Hard",
    "version": 1,
    "function_name": "supervisor_orchestration",
    "description_en": r"""Implement a deterministic supervisor that routes independent tasks to capability-declaring workers.

Signature: 'supervisor_orchestration(request, workers, scheduler, bus, limits, checkpoint_store, trace) -> OrchestrationResult'. Use 'MessageEnvelope', 'OrchestrationResult', 'RetryableToolError', and 'PermanentToolError' from 'torch_judge.harness.agents'.

'request' is:

'{"run_id":str, "correlation_id":str, "tasks":[{"task_id":str, "capability":str, "payload":dict, "deadline":number|None, "idempotency_key":str|None}, ...]}'

Task ids are unique. 'limits' has positive integer 'max_concurrency' and 'max_attempts', non-negative integer 'max_queue', and optional boolean/callable 'cancelled'. The checkpoint store exposes 'contains(run_id, task_id)', 'get(...)', and 'put(...)'. Checkpointed results—including JSON null—are restored before queue admission and never rerun. Admit at most 'max_queue' remaining tasks; reject overflow with a typed 'queue_overflow' failure.

Process admitted work in deterministic waves no larger than 'max_concurrency'. This simulates a concurrency gate—the harness remains synchronous and does not claim wall-clock parallelism. Ask 'scheduler.choose(capability, workers)' for a worker. Send an isolated envelope through 'bus.send(envelope)' before 'worker.run(envelope)'. The first envelope has 'attempt=0'; retries increment it. Derive stable message ids from run/task/attempt, preserve the request correlation and parent run ids, and reuse the task idempotency key (or derive 'run_id:task_id') across retries.

On worker success, checkpoint before publishing the result. A checkpoint write failure is an infrastructure failure and must not create contradictory success. Requeue retryable failures and 'TimeoutError' at the tail for fairness until 'max_attempts'; retry exhaustion and permanent errors become typed dead-letter failures. Preserve all durable successful partial results. Missing capabilities are typed failures. Cancellation stops dispatching new waves and marks pending tasks cancelled. Return status 'completed', 'partial', 'failed', or 'cancelled', plus results, failures, attempt counts, observed maximum wave size, and resumed ids. A 'HarnessFailure' denotes a broken evaluator fixture and must escape unchanged.

Emit 'orchestration.started', task lifecycle events, and exactly one terminal orchestration event. All events share the request run and correlation ids.

Optional code reading: OpenClaw 'createSessionsSpawnTool' validates isolated subagent/ACP spawn context and passes requester run identity into child creation. Hermes 'dispatch_once' coordinates durable Kanban claims, worker processes, retry limits, and idempotency in SQLite. Microsoft Agent Framework 'ConcurrentBuilder' explicitly wires dispatcher fan-out, participant fan-in, an aggregator, and optional checkpoint storage.""",
    "advisory_prerequisites": ["tool_registry", "budgeted_agent_loop"],
    "design_note_rubric": build_design_note_rubric(),
    "hints": [
        {"level": 1, "kind": "questions", "content": "Which work must be restored before applying queue pressure? What makes a retry envelope different—and what must remain identical? Where should a retry be placed for fairness? When can a checkpoint be written safely? How do partial results survive another task's failure?"},
        {"level": 2, "kind": "analysis", "content": "Validate and copy the request, restore checkpoints, then split remaining tasks into admitted and overflow. Pop at most max_concurrency items per wave. For each item choose an eligible worker, increment its attempt count, construct and send an envelope, and classify the outcome. Checkpoint success immediately; append retryable work to the queue tail; record every terminal failure without discarding results. Derive final status only after the queue is terminal or cancelled."},
    ],
    "model_connections": [
        "OpenClaw sessions_spawn carries requester run/session context, inherited tool policy, sandbox mode, and structured spawn failures into subagent or ACP runtimes.",
        "Hermes Kanban uses a durable SQLite board, per-task retry/idempotency fields, worker claims, heartbeats, crash detection, and dispatch passes rather than fragile in-process-only swarms.",
        "Microsoft Agent Framework ConcurrentBuilder wires dispatcher → fan-out participants → fan-in aggregator and accepts CheckpointStorage for resumable workflow state.",
    ],
    "pro_con_analysis": {
        "pros": [
            "A central supervisor gives one place to enforce global concurrency, queue pressure, retries, and observability.",
            "Per-task checkpoints preserve partial progress and prevent repeated completed side effects after restart.",
            "Typed correlated envelopes make routing and retry state inspectable across process or transport boundaries.",
        ],
        "cons": [
            "A central supervisor is a throughput and availability bottleneck unless replicated with coordinated state.",
            "Wave-based deterministic execution demonstrates admission semantics but not real async race behavior.",
            "Peer-to-peer delegation can reduce central load, but makes global budgets, termination, and debugging harder.",
        ],
    },
    "sources": [
        {"kind": "code", "url": "https://github.com/openclaw/openclaw", "commit": "40d31f34813c2a01284b097c0d0d785fbb173400", "path": "src/agents/tools/sessions-spawn-tool.ts", "symbol": "createSessionsSpawnTool", "license": "MIT", "adapted": "Structured spawn validation, requester/child run context, isolated task payload, runtime selection, and typed failure results.", "simplifications": "No ACP, channel/thread binding, attachments, sandbox inheritance, model override, visibility mode, or asynchronous child registry."},
        {"kind": "code", "url": "https://github.com/NousResearch/hermes-agent", "commit": "299e409f15aa5615a8a64be488580be92cda351e", "path": "hermes_cli/kanban_db.py", "symbol": "Task, DispatchResult, dispatch_once, detect_crashed_workers", "license": "MIT", "adapted": "Durable task identity, idempotency keys, retry limits, worker dispatch, crash classification, and partial board state.", "simplifications": "In-memory queue/checkpoints, synchronous workers, no SQLite claims, OS processes, heartbeats, workspace isolation, dependency graph, or stale-worker reclaim."},
        {"kind": "code", "url": "https://github.com/microsoft/agent-framework", "commit": "7c6b1e975f75193ace223a05c6535b8556f93ee4", "path": "python/packages/orchestrations/agent_framework_orchestrations/_concurrent.py", "symbol": "_DispatchToAllParticipants, _AggregateAgentConversations, ConcurrentBuilder.build", "license": "MIT", "adapted": "Explicit dispatcher fan-out, deterministic fan-in aggregation, participant execution, and optional checkpoint storage.", "simplifications": "No async Workflow/Executor graph, streaming, human request-info loop, participant output selection, or agent conversation objects."},
    ],
    "tests": [
        {"name": "Bounded waves, fair routing, and correlated envelopes", "behavior": "scheduler.concurrency", "code": r"""
from torch_judge.harness.agents import InMemoryBus,InMemoryCheckpointStore,RoundRobinScheduler,ScriptedWorker,TraceRecorder
workers=[ScriptedWorker('a',{'search'},[{'by':'a'}]),ScriptedWorker('b',{'search'},[{'by':'b'}]),ScriptedWorker('calc',{'math'},[{'value':4}])]
request={'run_id':'run','correlation_id':'corr','tasks':[
 {'task_id':'t1','capability':'search','payload':{'q':'x'},'deadline':5,'idempotency_key':'k1'},
 {'task_id':'t2','capability':'search','payload':{'q':'y'},'deadline':5,'idempotency_key':'k2'},
 {'task_id':'t3','capability':'math','payload':{'x':2},'deadline':None,'idempotency_key':None},
]}
bus=InMemoryBus(); store=InMemoryCheckpointStore(); trace=TraceRecorder()
r={fn}(request,workers,RoundRobinScheduler(),bus,{'max_concurrency':2,'max_attempts':2,'max_queue':3},store,trace)
assert r.status=='completed' and r.results=={'t1':{'by':'a'},'t2':{'by':'b'},'t3':{'value':4}}
assert r.attempts=={'t1':1,'t2':1,'t3':1} and r.max_in_flight==2 and store.puts==3
assert [(m.recipient,m.attempt,m.parent_run_id,m.correlation_id,m.idempotency_key) for m in bus.messages]==[
 ('a',0,'run','corr','k1'),('b',0,'run','corr','k2'),('calc',0,'run','corr','run:t3')]
assert len({m.message_id for m in bus.messages})==3
assert all(e.run_id=='run' and e.correlation_id=='corr' for e in trace.events)
"""},
        {"name": "Retries are fair and permanent failures are dead-lettered", "behavior": "retry.classification", "code": r"""
from torch_judge.harness.agents import InMemoryBus,InMemoryCheckpointStore,PermanentToolError,RetryableToolError,RoundRobinScheduler,ScriptedWorker,TraceRecorder
retry=ScriptedWorker('retry',{'api'},[RetryableToolError('busy'),{'ok':True}])
deny=ScriptedWorker('deny',{'write'},[PermanentToolError('forbidden')])
request={'run_id':'r','correlation_id':'c','tasks':[
 {'task_id':'a','capability':'api','payload':{},'deadline':None,'idempotency_key':'same'},
 {'task_id':'b','capability':'write','payload':{},'deadline':None,'idempotency_key':None},
]}
bus=InMemoryBus(); trace=TraceRecorder()
r={fn}(request,[retry,deny],RoundRobinScheduler(),bus,{'max_concurrency':1,'max_attempts':2,'max_queue':2},InMemoryCheckpointStore(),trace)
assert r.status=='partial' and r.results=={'a':{'ok':True}}
assert r.failures=={'b':{'type':'dead_letter','reason':'permanent','worker':'deny','message':'forbidden'}}
assert r.attempts=={'a':2,'b':1}
assert [m.message_id for m in bus.messages]==['r:a:0','r:b:0','r:a:1']
assert [m.attempt for m in retry.envelopes]==[0,1]
assert [m.idempotency_key for m in retry.envelopes]==['same','same']
"""},
        {"name": "Resume precedes queue admission and avoids duplicate effects", "behavior": "checkpoint.recovery", "visibility": "unshown", "failure_message": "Restore completed tasks before queue admission, preserve partial results, and never rerun checkpointed effects.", "code": r"""
from torch_judge.harness.agents import InMemoryBus,InMemoryCheckpointStore,RoundRobinScheduler,ScriptedWorker,TraceRecorder
store=InMemoryCheckpointStore(); store.put('r','done',{'saved':1})
worker=ScriptedWorker('w',{'job'},[{'fresh':2},{'overflow':3}])
tasks=[
 {'task_id':'done','capability':'job','payload':{},'deadline':None,'idempotency_key':'done'},
 {'task_id':'fresh','capability':'job','payload':{},'deadline':None,'idempotency_key':'fresh'},
 {'task_id':'overflow','capability':'job','payload':{},'deadline':None,'idempotency_key':'overflow'},
]
r={fn}({'run_id':'r','correlation_id':'c','tasks':tasks},[worker],RoundRobinScheduler(),InMemoryBus(),{'max_concurrency':1,'max_attempts':1,'max_queue':1},store,TraceRecorder())
assert r.status=='partial' and r.results=={'done':{'saved':1},'fresh':{'fresh':2}}
assert r.resumed==['done'] and r.attempts=={'done':0,'fresh':1,'overflow':0}
assert r.failures=={'overflow':{'type':'queue_overflow'}}
assert worker.calls==1 and worker.effects==1 and store.puts==2
"""},
        {"name": "Missing workers, timeouts, and retry exhaustion stay failures", "behavior": "protocol.validation", "visibility": "unshown", "failure_message": "Do not report unroutable, timed-out, or retry-exhausted tasks as successful results.", "code": r"""
from torch_judge.harness.agents import InMemoryBus,InMemoryCheckpointStore,RoundRobinScheduler,ScriptedWorker,TraceRecorder
timeout=ScriptedWorker('slow',{'slow'},[TimeoutError('late'),TimeoutError('late again')])
tasks=[
 {'task_id':'missing','capability':'none','payload':{},'deadline':None,'idempotency_key':None},
 {'task_id':'slow','capability':'slow','payload':{},'deadline':1,'idempotency_key':None},
]
r={fn}({'run_id':'r','correlation_id':'c','tasks':tasks},[timeout],RoundRobinScheduler(),InMemoryBus(),{'max_concurrency':2,'max_attempts':2,'max_queue':2},InMemoryCheckpointStore(),TraceRecorder())
assert r.status=='failed' and r.results=={}
assert r.failures['missing']=={'type':'no_worker','capability':'none'}
assert r.failures['slow']=={'type':'dead_letter','reason':'timeout','worker':'slow','attempts':2,'message':'late again'}
assert r.attempts=={'missing':0,'slow':2}
"""},
        {"name": "Cancellation stops later waves and preserves prior success", "behavior": "state.invariant", "visibility": "unshown", "failure_message": "Observe cancellation between waves, dispatch no new work, and retain already completed task results.", "code": r"""
from torch_judge.harness.agents import InMemoryBus,InMemoryCheckpointStore,RoundRobinScheduler,ScriptedWorker,TraceRecorder
bus=InMemoryBus(); worker=ScriptedWorker('w',{'job'},[{'one':1},{'two':2}])
tasks=[{'task_id':x,'capability':'job','payload':{},'deadline':None,'idempotency_key':x} for x in ('one','two')]
limits={'max_concurrency':1,'max_attempts':1,'max_queue':2,'cancelled':lambda: len(bus.messages)>=1}
r={fn}({'run_id':'r','correlation_id':'c','tasks':tasks},[worker],RoundRobinScheduler(),bus,limits,InMemoryCheckpointStore(),TraceRecorder())
assert r.status=='partial' and r.results=={'one':{'one':1}}
assert r.failures=={'two':{'type':'cancelled'}} and worker.calls==1
"""},
        {"name": "Unexpected worker failures are isolated and dead-lettered", "behavior": "retry.classification", "visibility": "unshown", "failure_message": "An unexpected worker or API adapter exception must become a typed task failure without crashing the orchestration or losing sibling success.", "code": r"""
from torch_judge.harness.agents import InMemoryBus,InMemoryCheckpointStore,RoundRobinScheduler,ScriptedWorker,TraceRecorder
good=ScriptedWorker('good',{'good'},[{'ok':1}]); bad=ScriptedWorker('bad',{'bad'},[RuntimeError('adapter broke')])
tasks=[
 {'task_id':'good','capability':'good','payload':{},'deadline':None,'idempotency_key':None},
 {'task_id':'bad','capability':'bad','payload':{},'deadline':None,'idempotency_key':None},
]
r={fn}({'run_id':'r','correlation_id':'c','tasks':tasks},[good,bad],RoundRobinScheduler(),InMemoryBus(),{'max_concurrency':2,'max_attempts':2,'max_queue':2},InMemoryCheckpointStore(),TraceRecorder())
assert r.status=='partial' and r.results=={'good':{'ok':1}}
assert r.failures=={'bad':{'type':'dead_letter','reason':'unexpected','worker':'bad','error_type':'RuntimeError','message':'adapter broke'}}
assert r.attempts=={'good':1,'bad':1}
"""},
        {"name": "A checkpointed null result is resumed", "behavior": "checkpoint.recovery", "visibility": "unshown", "failure_message": "Checkpoint presence must be tested independently from its JSON value so a completed null result is not rerun.", "code": r"""
from torch_judge.harness.agents import InMemoryBus,InMemoryCheckpointStore,RoundRobinScheduler,ScriptedWorker,TraceRecorder
store=InMemoryCheckpointStore(); store.put('r','done',None)
worker=ScriptedWorker('w',{'job'},[{'must':'not run'}])
task={'task_id':'done','capability':'job','payload':{},'deadline':None,'idempotency_key':None}
r={fn}({'run_id':'r','correlation_id':'c','tasks':[task]},[worker],RoundRobinScheduler(),InMemoryBus(),{'max_concurrency':1,'max_attempts':1,'max_queue':1},store,TraceRecorder())
assert r.status=='completed' and r.results=={'done':None}
assert r.resumed==['done'] and r.attempts=={'done':0}
assert worker.calls==0 and store.puts==1
"""},
        {"name": "Checkpoint write failure never publishes contradictory success", "behavior": "checkpoint.recovery", "visibility": "unshown", "failure_message": "Publish a result only after durable checkpoint success and classify storage failure separately from worker failure.", "code": r"""
from torch_judge.harness.agents import InMemoryBus,RoundRobinScheduler,ScriptedWorker,TraceRecorder
class FailingStore:
    def contains(self,*args): return False
    def get(self,*args): return None
    def put(self,*args): raise OSError('disk full')
task={'task_id':'t','capability':'job','payload':{},'deadline':None,'idempotency_key':'k'}
worker=ScriptedWorker('w',{'job'},[{'ok':1}])
r={fn}({'run_id':'r','correlation_id':'c','tasks':[task]},[worker],RoundRobinScheduler(),InMemoryBus(),{'max_concurrency':1,'max_attempts':1,'max_queue':1},FailingStore(),TraceRecorder())
assert r.status=='failed' and r.results=={}
assert r.failures=={'t':{'type':'checkpoint_failure','error_type':'OSError','message':'disk full'}}
assert worker.calls==1 and worker.effects==1
"""},
        {"name": "Worker harness exhaustion remains an evaluator error", "behavior": "state.invariant", "visibility": "unshown", "failure_message": "An exhausted scripted worker is an invalid evaluator fixture and must escape instead of becoming a learner-facing dead letter.", "code": r"""
from torch_judge.harness import HarnessFailure
from torch_judge.harness.agents import InMemoryBus,InMemoryCheckpointStore,RoundRobinScheduler,ScriptedWorker,TraceRecorder
task={'task_id':'t','capability':'job','payload':{},'deadline':None,'idempotency_key':None}
try:
    {fn}({'run_id':'r','correlation_id':'c','tasks':[task]},[ScriptedWorker('w',{'job'},[])],RoundRobinScheduler(),InMemoryBus(),{'max_concurrency':1,'max_attempts':1,'max_queue':1},InMemoryCheckpointStore(),TraceRecorder())
except HarnessFailure: pass
else: raise AssertionError('worker exhaustion must escape')
"""},
    ],
    "solution": r'''def supervisor_orchestration(request, workers, scheduler, bus, limits, checkpoint_store, trace):
    from collections import deque
    from copy import deepcopy
    from torch_judge.harness import HarnessFailure
    from torch_judge.harness.agents import MessageEnvelope, OrchestrationResult, PermanentToolError, RetryableToolError
    from torch_judge.harness.agents.protocol import copy_json

    if not isinstance(request, dict):
        raise ValueError("request must be a dictionary")
    run_id = request.get("run_id")
    correlation_id = request.get("correlation_id")
    tasks = request.get("tasks")
    if not isinstance(run_id, str) or not run_id or not isinstance(correlation_id, str) or not correlation_id:
        raise ValueError("run and correlation ids must be non-empty")
    if not isinstance(tasks, list):
        raise ValueError("tasks must be a list")
    tasks = deepcopy(tasks)
    seen = set()
    for task in tasks:
        if not isinstance(task, dict):
            raise ValueError("each task must be a dictionary")
        task_id = task.get("task_id")
        capability = task.get("capability")
        payload = task.get("payload")
        deadline = task.get("deadline")
        key = task.get("idempotency_key")
        if (not isinstance(task_id, str) or not task_id or task_id in seen
                or not isinstance(capability, str) or not capability
                or not isinstance(payload, dict)
                or (deadline is not None and (isinstance(deadline, bool)
                    or not isinstance(deadline, (int, float)) or deadline < 0))
                or (key is not None and (not isinstance(key, str) or not key))):
            raise ValueError("malformed or duplicate task")
        copy_json(payload, field="task payload JSON")
        seen.add(task_id)

    max_concurrency = limits.get("max_concurrency")
    max_attempts = limits.get("max_attempts")
    max_queue = limits.get("max_queue")
    if (not isinstance(max_concurrency, int) or isinstance(max_concurrency, bool) or max_concurrency < 1
            or not isinstance(max_attempts, int) or isinstance(max_attempts, bool) or max_attempts < 1
            or not isinstance(max_queue, int) or isinstance(max_queue, bool) or max_queue < 0):
        raise ValueError("invalid orchestration limits")

    def is_cancelled():
        value = limits.get("cancelled", False)
        return bool(value() if callable(value) else value)

    results = {}
    failures = {}
    attempts = {task["task_id"]: 0 for task in tasks}
    resumed = []
    remaining = []
    max_in_flight = 0
    cancelled_run = False
    trace.emit("orchestration.started", 0, run_id, correlation_id, {"tasks": len(tasks)})

    for task in tasks:
        if checkpoint_store.contains(run_id, task["task_id"]):
            checkpoint = checkpoint_store.get(run_id, task["task_id"])
            results[task["task_id"]] = checkpoint
            resumed.append(task["task_id"])
            trace.emit("task.resumed", 0, run_id, correlation_id, {"task_id": task["task_id"]})
        else:
            remaining.append(task)

    admitted = remaining[:max_queue]
    for task in remaining[max_queue:]:
        failures[task["task_id"]] = {"type": "queue_overflow"}
        trace.emit("queue.rejected", 0, run_id, correlation_id, {"task_id": task["task_id"]})
    pending = deque(admitted)

    while pending:
        if is_cancelled():
            cancelled_run = True
            while pending:
                task = pending.popleft()
                failures[task["task_id"]] = {"type": "cancelled"}
                trace.emit("task.cancelled", 0, run_id, correlation_id, {"task_id": task["task_id"]})
            break
        wave = [pending.popleft() for _ in range(min(max_concurrency, len(pending)))]
        max_in_flight = max(max_in_flight, len(wave))
        trace.emit("wave.started", 0, run_id, correlation_id, {"size": len(wave)})
        for task in wave:
            task_id = task["task_id"]
            worker = scheduler.choose(task["capability"], workers)
            if worker is None:
                failures[task_id] = {"type": "no_worker", "capability": task["capability"]}
                trace.emit("task.failed", 0, run_id, correlation_id,
                           {"task_id": task_id, "type": "no_worker"})
                continue
            attempt_index = attempts[task_id]
            attempts[task_id] += 1
            envelope = MessageEnvelope(
                f"{run_id}:{task_id}:{attempt_index}", correlation_id, run_id,
                "supervisor", worker.name, task["deadline"], attempt_index,
                task["idempotency_key"] or f"{run_id}:{task_id}", deepcopy(task["payload"]),
            )
            bus.send(envelope)
            trace.emit("task.dispatched", 0, run_id, correlation_id,
                       {"task_id": task_id, "worker": worker.name, "attempt": attempt_index})
            try:
                value = worker.run(envelope)
                value = copy_json(value, field="worker result JSON")
                try:
                    checkpoint_store.put(run_id, task_id, value)
                except HarnessFailure:
                    raise
                except Exception as error:
                    failures[task_id] = {"type": "checkpoint_failure",
                                         "error_type": type(error).__name__,
                                         "message": str(error)}
                    trace.emit("task.failed", 0, run_id, correlation_id,
                               {"task_id": task_id, "type": "checkpoint_failure"})
                else:
                    results[task_id] = value
                    trace.emit("task.completed", 0, run_id, correlation_id,
                               {"task_id": task_id, "worker": worker.name})
            except HarnessFailure:
                raise
            except PermanentToolError as error:
                failures[task_id] = {"type": "dead_letter", "reason": "permanent",
                                     "worker": worker.name, "message": str(error)}
                trace.emit("task.dead_letter", 0, run_id, correlation_id,
                           {"task_id": task_id, "reason": "permanent"})
            except (RetryableToolError, TimeoutError) as error:
                reason = "timeout" if isinstance(error, TimeoutError) else "retry_exhausted"
                if attempts[task_id] < max_attempts:
                    pending.append(task)
                    trace.emit("task.retry", 0, run_id, correlation_id,
                               {"task_id": task_id, "attempt": attempts[task_id]})
                else:
                    failures[task_id] = {"type": "dead_letter", "reason": reason,
                                         "worker": worker.name, "attempts": attempts[task_id],
                                         "message": str(error)}
                    trace.emit("task.dead_letter", 0, run_id, correlation_id,
                               {"task_id": task_id, "reason": reason})
            except Exception as error:
                failures[task_id] = {"type": "dead_letter", "reason": "unexpected",
                                     "worker": worker.name,
                                     "error_type": type(error).__name__,
                                     "message": str(error)}
                trace.emit("task.dead_letter", 0, run_id, correlation_id,
                           {"task_id": task_id, "reason": "unexpected"})

    if failures and results:
        status = "partial"
    elif failures:
        status = "cancelled" if cancelled_run else "failed"
    else:
        status = "completed"
    trace.emit(f"orchestration.{status}", 0, run_id, correlation_id,
               {"results": len(results), "failures": len(failures)})
    return OrchestrationResult(status, results, failures, attempts, max_in_flight, resumed)''',
    "demo": r"""from torch_judge.harness.agents import InMemoryBus,InMemoryCheckpointStore,RoundRobinScheduler,ScriptedWorker,TraceRecorder
request={'run_id':'demo','correlation_id':'demo-c','tasks':[{'task_id':'research','capability':'search','payload':{'q':'agents'},'deadline':None,'idempotency_key':None}]}
result=supervisor_orchestration(request,[ScriptedWorker('researcher',{'search'},[{'summary':'done'}])],RoundRobinScheduler(),InMemoryBus(),{'max_concurrency':1,'max_attempts':2,'max_queue':4},InMemoryCheckpointStore(),TraceRecorder())
print(result.status,result.results)""",
}
