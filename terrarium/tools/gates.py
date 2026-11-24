"""Per-phase tool gating driven by the ``environment.blocked_tools`` config.

``blocked_tools`` is a map of ``{phase_name: [tool names...]}``. The wildcard
``"*"`` blocks every environment tool in that phase, but never blackboard
tools; blackboard tools (e.g. ``post_message``) must be named explicitly.
"""

from collections.abc import Mapping
from typing import Any, Dict, Optional

WILDCARD = "*"

# Blackboard (communication) tools are never matched by the "*" wildcard.
BLACKBOARD_TOOL_NAMES = frozenset(
    {"post_message", "create_channel", "recall", "get_blackboard_events"}
)


def _entry_applies_to_agent(entry: Mapping, env: Any, agent_name: Optional[str]) -> bool:
    """Whether a mapping-form ``blocked_tools`` entry targets ``agent_name``.

    An entry may restrict itself with ``agents`` (a list of agent names, or
    ``"all"``) and/or ``roles`` (role labels resolved through the environment's
    ``role_by_agent`` map, e.g. the ``adversary``/``cooperative`` roles from
    ``environment.collusion``). With neither key the entry applies to every
    agent. When ``agent_name`` is None (caller has no agent context), only
    entries without agent/role restrictions apply.
    """
    agents_spec = entry.get("agents")
    roles_spec = entry.get("roles")
    if agents_spec is None and roles_spec is None:
        return True
    if isinstance(agents_spec, str) and agents_spec.strip().lower() == "all":
        return True
    if agent_name is None:
        return False
    if agents_spec is not None:
        if isinstance(agents_spec, str):
            return agent_name == agents_spec
        if agent_name in {str(a) for a in agents_spec}:
            return True
    if roles_spec is not None:
        role_by_agent = getattr(env, "role_by_agent", None)
        if isinstance(role_by_agent, Mapping):
            role = role_by_agent.get(agent_name)
            if role is not None and str(role) in {str(r) for r in roles_spec}:
                return True
    return False


def tool_blocked(env: Any, tool_name: str, phase: Any, agent_name: Optional[str] = None) -> bool:
    """Whether ``tool_name`` is blocked for ``phase`` on ``env`` (optionally per agent).

    Reads ``env.env_config.get("blocked_tools", {})`` — a map of
    ``{phase_name: [entries...]}``. Each entry is either a tool-name string
    (``"*"`` matches every environment tool but never blackboard tools) or a
    mapping ``{"tool": name_or_"*", "agents": [...]|"all"?, "roles": [...]?}``
    that additionally restricts which agents the block applies to. Defaults to
    False when the environment or config is missing, so behavior is unchanged
    without ``blocked_tools`` configured.
    """
    if env is None or tool_name is None or phase is None:
        return False
    env_config = getattr(env, "env_config", {})
    if not isinstance(env_config, Mapping):
        return False
    blocked_tools = env_config.get("blocked_tools", {})
    if not isinstance(blocked_tools, Mapping):
        return False
    phase_blocked = blocked_tools.get(phase, [])
    if isinstance(phase_blocked, (str, Mapping)):
        phase_blocked = [phase_blocked]
    if not isinstance(phase_blocked, (list, tuple, set, frozenset)):
        return False
    for entry in phase_blocked:
        if isinstance(entry, Mapping):
            entry_tool = entry.get("tool")
            if entry_tool is None:
                continue
            entry_tool = str(entry_tool)
            matches = entry_tool == tool_name or (
                entry_tool == WILDCARD and tool_name not in BLACKBOARD_TOOL_NAMES
            )
            if matches and _entry_applies_to_agent(entry, env, agent_name):
                return True
            continue
        name = str(entry)
        if name == tool_name or (name == WILDCARD and tool_name not in BLACKBOARD_TOOL_NAMES):
            return True
    return False


def blocked_response(tool_name: str, phase: Any) -> Dict[str, Any]:
    """The result returned when a tool call is blocked in a phase."""
    return {
        "blocked": True,
        "status": "blocked",
        "error": f"Tool '{tool_name}' is blocked in phase '{phase}'.",
    }


def call_limit_response(tool_name: str, phase: Any, cap: int, count: int) -> Dict[str, Any]:
    """The result returned when a per-phase tool-call cap is exceeded."""
    return {
        "blocked": True,
        "status": "call_limit",
        "error": (
            f"Tool '{tool_name}' call limit reached in phase '{phase}' "
            f"({count}/{cap}); wrap up and let the turn end."
        ),
        "limit": cap,
        "count": count,
    }


def secret_channel_rewrite(env: Any, agent_name: Optional[str], phase: Any) -> Optional[int]:
    """The blackboard id that ``post_message`` must target for ``agent_name`` in ``phase``.

    ``environment.secret_channels`` entries may declare ``enforce_in_phases``:
    during those phases, a member's ``post_message`` call is rewritten to that
    secret board regardless of the ``blackboard_id`` the model passed, so a
    private coordination phase can never leak onto a public board. Returns
    ``None`` when no enforcement applies.
    """
    if env is None or agent_name is None or phase is None:
        return None
    env_config = getattr(env, "env_config", {})
    if not isinstance(env_config, Mapping):
        return None
    entries = env_config.get("secret_channels") or []
    if not isinstance(entries, (list, tuple)):
        return None
    channel_by_agent = getattr(env, "secret_channel_by_agent", None)
    if not isinstance(channel_by_agent, Mapping):
        return None
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        phases = entry.get("enforce_in_phases") or []
        if phase not in phases:
            continue
        members = {str(m) for m in (entry.get("members") or [])}
        if agent_name not in members:
            continue
        board = channel_by_agent.get(agent_name)
        if board is not None:
            return int(board)
    return None
