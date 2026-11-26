# Design: per-phase tool policy + native local protocol (pre-coordination support)

Status: **proposal.** Companion to the Colosseum-side plan
(`colosseum-collusion-via-tools: docs/terrarium-integration.md`). Goal: make
the collusion-via-tools pre-coordination condition expressible natively in
Terrarium as *data* (a per-phase tool policy on the protocol instance),
replacing the downstream monkey-patch in Colosseum's
`experiments/common/_patch_terrarium.py` and its `contextvars` flags.

Defaults must not change: with no policy set, behavior is exactly today's.

## Motivation

Today, `ToolsetDiscovery.get_tools_for_blackboard` enables blackboard tools
only for `phase == "planning"` (`terrarium/tools/discovery.py:85`), and
Colosseum patches it to (a) remap `pre_coordination -> planning` and (b)
return `[]` when `post_message` is disabled. The patch exists because the
flag had to live in process-global state. But `ToolsetDiscovery` is
**per-agent** (`terrarium/agents/base.py:55`) and the phase is known when the
per-turn tool list is built (`terrarium/agents/base.py:403-412`), and the
communication protocol is per-run — so the gating can be explicit per-phase
data with no patching at all.

## Proposed API

### 1. `terrarium/tools/discovery.py`

- A planning-shaped phase set, e.g. `ToolsetDiscovery.planning_shaped_phases:
  Set[str] = {"planning"}` (class attribute, overridable per instance or via
  config), used by `get_tools_for_blackboard` instead of the hardcoded
  `phase == "planning"` check. `"pre_coordination"` is added by the runner
  that wants it (config-declared, keeping Terrarium generic).
- `allowed_tools: Optional[Set[str]] = None` parameter on
  `get_tools_for_blackboard` and `get_blackboard_tool_names`: when given, the
  result is intersected with it (`set()` withholds all blackboard tools —
  the `disable_post_message` axis).

### 2. `terrarium/communication_protocols/local.py` (new)

Move `LocalCommunicationProtocol` up from Colosseum's
`experiments/common/local_protocol.py`, refactored around an explicit policy:

```python
@dataclass
class PhaseToolPolicy:
    environment_tools_offered: bool = True        # schemas still shown
    block_environment_tool_calls: bool = False    # calls rejected at runtime
    blackboard_tools: Optional[Set[str]] = None   # None = phase default
    route_post_message_to: Optional[int] = None   # secret board id
    max_post_messages_per_agent: Optional[int] = None  # None = unlimited
```

Protocol API: `set_phase_policy(phase, policy)` / `pop_phase_policy(phase)`,
held in a per-instance dict (no contextvars — each run has its own protocol,
so concurrent runs cannot clobber each other). Enforcement points, mirroring
today's patched behavior:

- `environment_handle_tool_call`: if the active phase policy blocks env
  calls, return `{"blocked": True, ...}` without executing.
- `blackboard_handle_tool_call`: if `route_post_message_to` is set, rewrite
  `post_message`'s `blackboard_id`; enforce `max_post_messages_per_agent`
  via a per-agent counter with `reset_post_message_counts()` at round
  boundaries (defense in depth: limit 0 rejects all calls).
- Keep the generic `agent_turn` dispatch used by `run_phase_entries`,
  `generate_comm_network` (coalition-only secret board creation), and
  secret-board scrubbing so non-participants never see the coalition board.

### 3. `terrarium/agents/base.py`

At the per-turn tool-list build (`base.py:403-412`), pass the active phase
policy's `blackboard_tools` as `allowed_tools` into
`get_tools_for_blackboard`. The protocol (which sets `agent.current_phase`
via `set_meta_context`) also exposes the active policy, e.g.
`agent.active_phase_policy`, so the agent needs no global state.

### 4. `terrarium/environments/dcops/hospital/hospital_env.py`

Native phase hook in `build_agent_context` (`hospital_env.py:561`, already
receives `phase` and `round_number`): `if phase == "pre_coordination":`
return the minimal identity + coalition context (no inventory, failures,
queue, or costs), replacing Colosseum's `_pre_coordination_context` scrub.

### 5. Tests

- Repoint `terrarium/tests/test_local_protocol.py` from
  `experiments.common.local_protocol` to `terrarium.communication_protocols.local`
  (removes the backwards dependency on the downstream repo).
- New tests: planning-shaped phase set, `allowed_tools` filtering,
  policy push/pop, board routing, per-agent cap + per-round reset, blocked
  env calls, hospital `pre_coordination` context shape.
- Assert trajectory phase labels stay verbatim (`"pre_coordination"`) so
  downstream judges/metrics are unaffected.

## Out of scope

- `terrarium/attacks/`: the attacks framework stays payload-driven injection;
  this mechanism is run structure, not an injected payload.
- No new environment class (`HospitalCollusionEnvironment`): the condition is
  orthogonal to hospital game state.
- No PyPI release yet; Colosseum consumes this fork via the editable
  submodule pin.
