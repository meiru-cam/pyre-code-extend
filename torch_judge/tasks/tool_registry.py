"""Provider-neutral discovery, validation, and dispatch for agent tools."""

from torch_judge.tasks._schema import build_design_note_rubric

TASK = {
    "title": "Validated Agent Tool Registry",
    "difficulty": "Medium",
    "version": 1,
    "function_name": "ToolRegistry",
    "description_en": r"""Implement a small provider-neutral registry that owns tool discovery and invocation.

Define a 'ToolRegistry' class with:

- 'register(tool)' — a tool exposes non-empty 'name' and 'description', an 'argument_schema' mapping argument names to Python types, and 'invoke(arguments, idempotency_key=None)'. Reject duplicate names and unsupported schema types.
- 'describe()' — return tools in registration order as fresh dictionaries containing only 'name', 'description', and 'parameters'. Convert 'str', 'int', 'float', 'bool', 'list', and 'dict' to 'string', 'integer', 'number', 'boolean', 'array', and 'object'.
- 'invoke(name, arguments, idempotency_key=None)' — reject unknown tools and require an exact argument set with exact declared Python types before calling the tool. In particular, 'bool' is not an 'int'. Forward the idempotency key unchanged.

This exercise separates the model-facing schema from executable handlers. That boundary lets runtimes filter what the model can discover and validate requests before side effects. It deliberately omits optional arguments and full JSON Schema so the registry semantics remain the focus.

Optional code reading: OpenClaw's 'createOpenClawTools' constructs a per-run inventory, merges plugin tools, filters client capabilities, and wraps hooks. Hermes 'ToolRegistry' owns schema retrieval and dispatch for self-registering built-in tools. Read those exact sections after attempting the exercise if you want to compare a compact registry with production policy and availability layers.""",
    "advisory_prerequisites": [],
    "design_note_rubric": build_design_note_rubric(),
    "hints": [
        {"level": 1, "kind": "questions", "content": "Which state must preserve registration order? What should describe expose—and what should it hide? Can you validate all argument names and types before invoking anything? Why does isinstance(True, int) need special handling?"},
        {"level": 2, "kind": "analysis", "content": "Keep a name-to-tool dictionary; normal dictionaries preserve insertion order. Build each description from a fixed type-name mapping and return newly allocated dictionaries. At invoke time compare key sets first, then require type(value) is declared_type for every argument, and only then delegate with the unchanged idempotency key."},
    ],
    "model_connections": [
        "OpenClaw createOpenClawTools builds a run-specific tool inventory, adds plugin tools, filters by client capabilities, and applies before-tool-call wrappers; this exercise isolates its discovery/dispatch core.",
        "Hermes tools.registry.ToolRegistry centralizes registered schemas, availability checks, and dispatch; its production implementation also handles async tools, plugins, result normalization, and error sanitization.",
        "Provider-neutral descriptions create an adapter seam: an OpenAI-, Anthropic-, or local-model adapter can translate one internal contract without changing tool handlers.",
    ],
    "pro_con_analysis": {
        "pros": [
            "Central validation prevents malformed calls from reaching side-effecting handlers.",
            "A least-data discovery surface reduces accidental leakage of handler and runtime internals.",
            "A provider-neutral registry keeps model API formats outside domain tools.",
        ],
        "cons": [
            "A central registry can become a policy bottleneck and single coordination point.",
            "The restricted type map cannot express optional fields, unions, bounds, or nested JSON Schema.",
            "Registration order is deterministic but may couple prompt order to startup order.",
        ],
    },
    "sources": [
        {"kind": "code", "url": "https://github.com/openclaw/openclaw", "commit": "40d31f34813c2a01284b097c0d0d785fbb173400", "path": "src/agents/openclaw-tools.ts", "symbol": "createOpenClawTools and filterToolsByClientCaps", "license": "MIT", "adapted": "Per-run tool inventory construction, plugin composition, capability filtering, and wrapper boundary.", "simplifications": "No channel context, plugins, allow/deny policy, sandbox capabilities, hooks, or dynamic media tools."},
        {"kind": "code", "url": "https://github.com/NousResearch/hermes-agent", "commit": "299e409f15aa5615a8a64be488580be92cda351e", "path": "tools/registry.py", "symbol": "ToolRegistry.register, get_definitions, and dispatch", "license": "MIT", "adapted": "Central schema registration, sorted discovery concepts, handler lookup, and dispatch boundary.", "simplifications": "No discovery imports, toolsets, availability cache, plugin override policy, async bridge, locks, result normalization, or error sanitizer."},
    ],
    "tests": [
        {"name": "Discovery is ordered and least-data", "behavior": "protocol.validation", "code": r"""
from torch_judge.harness.agents import FakeTool
registry={fn}()
registry.register(FakeTool('search','find documents',{'q':str,'limit':int},outcomes=[{'items':[]}]))
registry.register(FakeTool('notify','send notice',{'urgent':bool},outcomes=[{'sent':True}]))
expected=[
 {'name':'search','description':'find documents','parameters':{'q':'string','limit':'integer'}},
 {'name':'notify','description':'send notice','parameters':{'urgent':'boolean'}},
]
actual=registry.describe()
assert actual==expected
actual[0]['parameters']['q']='changed'
assert registry.describe()==expected
"""},
        {"name": "Valid calls dispatch and preserve idempotency", "behavior": "effects.idempotency", "code": r"""
from torch_judge.harness.agents import FakeTool
tool=FakeTool('charge','charge once',{'amount':int},outcomes=[{'receipt':'r1'},{'receipt':'r2'}])
registry={fn}(); registry.register(tool)
assert registry.invoke('charge',{'amount':3},'pay-1')=={'receipt':'r1'}
assert registry.invoke('charge',{'amount':3},'pay-1')=={'receipt':'r1'}
assert tool.calls==2 and tool.effects==1
"""},
        {"name": "Invalid calls never reach a handler", "behavior": "protocol.validation", "visibility": "unshown", "failure_message": "Reject unknown, missing, extra, and type-invalid arguments before the tool handler runs.", "code": r"""
from torch_judge.harness.agents import FakeTool
tool=FakeTool('count','count',{'value':int},outcomes=[{'ok':True}])
registry={fn}(); registry.register(tool)
bad=[('missing',{},None),('extra',{'value':1,'x':2},None),('type',{'value':'1'},None),('bool-is-int',{'value':True},None)]
for _,arguments,key in bad:
    try: registry.invoke('count',arguments,key)
    except (TypeError,ValueError): pass
    else: raise AssertionError('invalid arguments were accepted')
try: registry.invoke('unknown',{},None)
except (KeyError,ValueError): pass
else: raise AssertionError('unknown tool was accepted')
assert tool.calls==0 and tool.effects==0
"""},
        {"name": "Registration rejects collisions and unsupported schemas", "behavior": "state.invariant", "visibility": "unshown", "failure_message": "Keep tool names unique and reject schema types the discovery format cannot represent.", "code": r"""
from torch_judge.harness.agents import FakeTool
registry={fn}(); registry.register(FakeTool('same','first',{},outcomes=[{}]))
for tool in [FakeTool('same','second',{},outcomes=[{}]),FakeTool('bad','bad',{'x':set},outcomes=[{}])]:
    try: registry.register(tool)
    except (TypeError,ValueError): pass
    else: raise AssertionError('invalid registration was accepted')
assert registry.describe()==[{'name':'same','description':'first','parameters':{}}]
"""},
    ],
    "solution": r'''class ToolRegistry:
    _TYPE_NAMES = {
        str: "string", int: "integer", float: "number",
        bool: "boolean", list: "array", dict: "object",
    }

    def __init__(self):
        self._tools = {}

    def register(self, tool):
        if not isinstance(tool.name, str) or not tool.name:
            raise ValueError("tool name must be non-empty")
        if tool.name in self._tools:
            raise ValueError(f"duplicate tool: {tool.name}")
        if not isinstance(tool.description, str) or not tool.description:
            raise ValueError("tool description must be non-empty")
        if not isinstance(tool.argument_schema, dict):
            raise TypeError("argument schema must be a dictionary")
        if any(not isinstance(name, str) or declared not in self._TYPE_NAMES
               for name, declared in tool.argument_schema.items()):
            raise TypeError("argument schema contains an unsupported entry")
        self._tools[tool.name] = tool

    def describe(self):
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": {
                    name: self._TYPE_NAMES[declared]
                    for name, declared in tool.argument_schema.items()
                },
            }
            for tool in self._tools.values()
        ]

    def invoke(self, name, arguments, idempotency_key=None):
        if name not in self._tools:
            raise KeyError(f"unknown tool: {name}")
        if not isinstance(arguments, dict):
            raise TypeError("arguments must be a dictionary")
        tool = self._tools[name]
        if set(arguments) != set(tool.argument_schema):
            raise ValueError("argument names do not match the schema")
        for argument, declared in tool.argument_schema.items():
            if type(arguments[argument]) is not declared:
                raise TypeError(f"invalid type for {argument}")
        return tool.invoke(arguments, idempotency_key=idempotency_key)''',
    "demo": r"""from torch_judge.harness.agents import FakeTool
registry=ToolRegistry()
registry.register(FakeTool('lookup','look up a record',{'id':str},outcomes=[{'name':'Ada'}]))
print(registry.describe())
print(registry.invoke('lookup',{'id':'42'},idempotency_key='lookup-42'))""",
}
