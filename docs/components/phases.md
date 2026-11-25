---
title: Phases
---

# Phases

The `simulation.phases` key defines what happens each iteration, in order: a list of phase entries, each running `rounds` rounds over all agents. It is the per-phase configuration toolkit — order and round count, tool gating, and per-phase prompts all live on the phase entry.

```yaml
simulation:
  max_iterations: 1
  seed: 42
  phases:
    - name: planning
      rounds: 2
    - name: critique
      rounds: 1
      prompt_file: prompts/critique.txt        # optional
      prompt_hook: my_package.hooks:critique_hook  # optional
      blocked_tools: ["schedule_patient"]      # optional blocklist
    - name: execution
      rounds: 1
```

When `phases` is absent, the sequence falls back to `[{name: planning, rounds: max_planning_rounds or 1}, {name: execution, rounds: 1}]`, so existing configs behave exactly as before.

Each entry supports:

- `name` (string, required) — the phase name. It is passed to `environment.build_agent_context(..., phase=name)`, to `protocol.agent_turn(..., phase=name, phase_config=entry)`, and to the agent, so environments can key prompt content off it (e.g. `hospital_prompts.py` switches on `agent_context["phase"]`).
- `rounds` (positive int, required) — how many rounds over all agents run in this phase, per iteration.
- `blocked_tools` (list of strings, optional) — tools disabled during this phase. See below.
- `prompt_file` (string, optional) — a text file whose contents are appended to every agent's user prompt during this phase, wrapped in a `=== PHASE INSTRUCTIONS (<name>) ===` block. Relative paths resolve against the config file's directory (the runner loads them once at startup via `load_phase_prompt_assets`).
- `prompt_hook` (string, optional) — a `"module:callable"` spec, imported at startup. On each agent turn in this phase the callable is invoked and its return string is appended to the user prompt after the `prompt_file` text. The hook may declare any of `agent_name`, `agent_context`, `blackboard_context`, `environment`, `phase`, `iteration`, `round_num` as parameters (or take `**kwargs`); only the declared ones are passed.

Unknown phase names are valid: they get the planning (blackboard-only) toolset and the execution tool-instruction block by default, so a custom phase works without touching environment code — the optional keys above let you shape it from the config alone.

## Tool gating (`blocked_tools`)

`blocked_tools` is a blocklist of tool names for that phase:

- `"*"` blocks **all environment tools** in that phase. It never blocks blackboard tools.
- `"post_message"` (named explicitly) blocks the blackboard messaging tool in that phase. Other blackboard tools (`create_channel`, `recall`, `get_blackboard_events`) can be listed the same way.
- Any other entry blocks that environment tool by name (e.g. `"schedule_patient"`).
- Phases without `blocked_tools` are unaffected.

### What happens when a blocked tool is attempted

- **Schema withholding**: the agent's tool list omits blocked tools (`BaseAgent` filters the environment and blackboard toolsets per phase), and environments can also withhold schemas in their tools class (the hospital does, in `HospitalTools.get_tools`).
- **Runtime enforcement**: even if a blocked tool is called anyway, the call is not executed and returns:

```python
{"blocked": True, "status": "blocked", "error": "Tool 'schedule_patient' is blocked in phase 'critique'."}
```

Environment tool calls are gated in the wrapper returned by `instantiate_environment_tools` (`terrarium/tools/environment.py`) and in `BaseAgent._execute_tool_call`; blackboard tool calls are gated in `SequentialCommunicationProtocol.blackboard_handle_tool_call`. The check itself lives in `terrarium/tools/gates.py` (`tool_blocked_in` / `tool_blocked`), which reads the `blocked_tools` list of the matching `simulation.phases` entry.

## Example

```yaml
simulation:
  max_iterations: 3
  seed: 42
  phases:
    - name: planning
      rounds: 2
    - name: pre_coordination
      rounds: 1
      blocked_tools: ["*", "post_message"]  # no env tools, no blackboard posting
    - name: execution
      rounds: 1
      blocked_tools: ["transfer_resources"] # withhold one tool in execution only
```
