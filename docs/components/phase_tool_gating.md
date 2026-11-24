---
title: Phase Tool Gating
---

# Phase Tool Gating

`environment.blocked_tools` disables tools per simulation phase, without monkeypatching. It maps a phase name to a list of blocked tool names:

```yaml
environment:
  blocked_tools:
    pre_coordination: ["*", "post_message"]
```

- `"*"` blocks **all environment tools** in that phase. It never blocks blackboard tools.
- `"post_message"` (named explicitly) blocks the blackboard messaging tool in that phase. Other blackboard tools (`create_channel`, `recall`, `get_blackboard_events`) can be listed the same way.
- Any other entry blocks that environment tool by name (e.g. `"schedule_patient"`).
- An entry may also be a mapping `{"tool": name_or_"*", "agents": [...], "roles": [...]}` to scope the block to specific agents: `agents` is a list of agent names (or `"all"`), `roles` resolves through the environment's `role_by_agent` map (e.g. `adversary` from `environment.collusion`). With neither key, the entry applies to every agent.
- Phases not listed are unaffected. With no `blocked_tools` key, behavior is exactly as before.

## What happens when a blocked tool is attempted

- **Schema withholding**: environments that support it (currently the hospital environment) omit blocked tools from `get_tools(phase)`, so the model never sees them. Blocking `"*"` yields an empty tool list for that phase.
- **Runtime enforcement**: even if a blocked tool is called anyway, the call is not executed and returns:

```python
{"blocked": True, "status": "blocked", "error": "Tool 'schedule_patient' is blocked in phase 'pre_coordination'."}
```

Environment tool calls are gated in the wrapper returned by `instantiate_environment_tools` (`terrarium/tools/environment.py`); blackboard tool calls are gated in `SequentialCommunicationProtocol.blackboard_handle_tool_call`. The check itself lives in `terrarium/tools/gates.py::tool_blocked(env, tool_name, phase)`.

## Example

```yaml
environment:
  blocked_tools:
    pre_coordination: ["*", "post_message"]   # no env tools, no blackboard posting
    planning: ["schedule_patient"]             # withhold one tool in planning only
    execution:
      - tool: post_message
        roles: [adversary]                     # adversary-only: post_message gone in execution
```

## Per-agent schema withholding

When the agent's `ToolsetDiscovery` is bound to the environment (the sequential protocol does this on every `agent_turn`), the tool schemas offered to a specific agent withhold the tools blocked for that agent in the current phase — so a blocked `post_message` is not even visible to the model. A stray call is still rejected at runtime with the `blocked` result above.

## Secret-channel enforcement

`environment.secret_channels` entries accept an optional `enforce_in_phases` list. During those phases, a member's `post_message` call is rewritten to that secret board no matter which `blackboard_id` the model passed, so a private coordination phase cannot leak onto a public board:

```yaml
environment:
  secret_channels:
    - members: [General_Hospital_Surgery, Resource_Provisioner]
      enforce_in_phases: [pre_coordination]
```
