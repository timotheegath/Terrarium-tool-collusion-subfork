"""Per-phase tool gating driven by ``simulation.phases[*].blocked_tools``.

Each entry of ``simulation.phases`` may list ``blocked_tools: [tool names...]``.
The wildcard ``"*"`` blocks every environment tool in that phase, but never
blackboard tools; blackboard tools (e.g. ``post_message``) must be named
explicitly. With no ``blocked_tools`` on a phase, nothing is blocked.
"""

from collections.abc import Mapping
from typing import Any, Dict, Optional, Set

WILDCARD = "*"

# Blackboard (communication) tools are never matched by the "*" wildcard.
BLACKBOARD_TOOL_NAMES = frozenset(
    {"post_message", "create_channel", "recall", "get_blackboard_events"}
)


def blocked_tools_for_phase(simulation_config: Optional[Mapping[str, Any]], phase: Any) -> Set[str]:
    """The set of blocked tool names for ``phase`` from ``simulation.phases``.

    Matches the phase entry whose ``name`` equals ``phase`` (exact, case-sensitive
    — ``resolve_phases`` strips whitespace). Returns an empty set when there is
    no ``phases`` key, no matching entry, or the entry has no ``blocked_tools``.
    """
    if not isinstance(simulation_config, Mapping) or phase is None:
        return set()
    raw_phases = simulation_config.get("phases")
    if not isinstance(raw_phases, list):
        return set()
    for entry in raw_phases:
        if not isinstance(entry, Mapping):
            continue
        if entry.get("name") != phase:
            continue
        blocked = entry.get("blocked_tools", [])
        if isinstance(blocked, str):
            blocked = [blocked]
        if not isinstance(blocked, (list, tuple, set, frozenset)):
            return set()
        return {str(name) for name in blocked}
    return set()


def tool_blocked_in(simulation_config: Optional[Mapping[str, Any]], tool_name: str, phase: Any) -> bool:
    """Whether ``tool_name`` is blocked for ``phase`` given a ``simulation`` config section."""
    if tool_name is None or phase is None:
        return False
    names = blocked_tools_for_phase(simulation_config, phase)
    if not names:
        return False
    if tool_name in names:
        return True
    return WILDCARD in names and tool_name not in BLACKBOARD_TOOL_NAMES


def tool_blocked(env: Any, tool_name: str, phase: Any) -> bool:
    """Whether ``tool_name`` is blocked for ``phase`` on ``env``.

    Reads ``env.full_config["simulation"]["phases"]`` (see
    ``simulation.phases[*].blocked_tools``). Defaults to False when the
    environment or config is missing, so behavior is unchanged without
    configured phases.
    """
    if env is None:
        return False
    full_config = getattr(env, "full_config", None)
    if not isinstance(full_config, Mapping):
        return False
    return tool_blocked_in(full_config.get("simulation"), tool_name, phase)


def blocked_response(tool_name: str, phase: Any) -> Dict[str, Any]:
    """The result returned when a tool call is blocked in a phase."""
    return {
        "blocked": True,
        "status": "blocked",
        "error": f"Tool '{tool_name}' is blocked in phase '{phase}'.",
    }
