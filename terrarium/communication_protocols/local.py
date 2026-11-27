from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Set

from terrarium.core.blackboard import Megaboard, format_blackboard_events_for_prompt
from terrarium.communication_protocols.base import BaseCommunicationProtocol
from terrarium.tools.environment import (
    EnvironmentToolsNotFoundError,
    call_env_tool,
    instantiate_environment_tools,
)

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class PhaseToolPolicy:
    """Per-phase tool policy held as explicit data on the protocol instance.

    Attributes:
        environment_tools_offered: Whether environment tool schemas are still
            shown to the agent during the phase.
        block_environment_tool_calls: Whether environment tool calls are
            rejected at runtime (the schema may still be offered).
        blackboard_tools: Allowlist of blackboard tool names for the phase
            (`None` = phase default; `set()` withholds all of them).
        route_post_message_to: Blackboard id that `post_message` calls are
            rewritten to (`None` = no rewriting).
        max_post_messages_per_agent: Per-agent `post_message` cap
            (`None` = unlimited; 0 rejects all calls).
    """

    environment_tools_offered: bool = True
    block_environment_tool_calls: bool = False
    blackboard_tools: Optional[Set[str]] = None
    route_post_message_to: Optional[int] = None
    max_post_messages_per_agent: Optional[int] = None


@dataclass(slots=True)
class ToolEvent:
    tool_name: str
    agent_name: str
    arguments: Dict[str, Any]
    result: Dict[str, Any]
    phase: Optional[str]
    iteration: Optional[int]
    planning_round: Optional[int]


class LocalCommunicationProtocol(BaseCommunicationProtocol):
    """In-process protocol (no MCP server) backed by Megaboard + environment-specific tools."""

    def __init__(
        self, *, config: Dict[str, Any], megaboard: Optional[Megaboard] = None
    ):
        self.config = config
        self.megaboard = megaboard or Megaboard()
        self.environment = None
        self.current_planning_round: Optional[int] = None
        self.tool_events: List[ToolEvent] = []

        # Environment tools implementation (local, no server).
        self._env_tools: Any = None
        self._env_tools_env_name: Optional[str] = None
        self._env_tools_env: Any = None

        # Per-phase tool policies and per-agent post_message counters. Both
        # live on the instance (one protocol per run), so concurrent runs
        # cannot clobber each other's policies.
        self._phase_policies: Dict[str, PhaseToolPolicy] = {}
        self._post_message_counts: Dict[str, int] = {}

    def set_phase_policy(self, phase: str, policy: PhaseToolPolicy) -> None:
        """Attach a tool policy to a phase label (e.g. "pre_coordination")."""
        self._phase_policies[phase] = policy

    def pop_phase_policy(self, phase: str) -> Optional[PhaseToolPolicy]:
        """Remove and return the policy attached to ``phase`` (None if unset)."""
        return self._phase_policies.pop(phase, None)

    def active_phase_policy(self, phase: Optional[str]) -> Optional[PhaseToolPolicy]:
        """Return the policy attached to ``phase`` (None when no policy is set)."""
        if phase is None:
            return None
        return self._phase_policies.get(phase)

    def reset_post_message_counts(self) -> None:
        """Reset the per-agent `post_message` counters (call at round boundaries)."""
        self._post_message_counts = {}

    def get_post_message_count(self, agent_name: str) -> int:
        """Return how many times `agent_name` has called `post_message` since the last reset."""
        return self._post_message_counts.get(agent_name, 0)

    def _record_post_message_call(self, agent_name: str) -> int:
        self._post_message_counts[agent_name] = (
            self._post_message_counts.get(agent_name, 0) + 1
        )
        return self._post_message_counts[agent_name]

    def _get_env_tools(self, environment: Any) -> Any:
        env_name = ""
        if environment is not None:
            # Allow environment subclasses to opt into a base toolset without requiring a
            # duplicate `*Tools` class to exist under terrarium/environments/**/*_tools.py.
            env_name = str(
                getattr(environment, "tools_environment_name", None)
                or environment.__class__.__name__
                or ""
            )
        if not env_name:
            raise EnvironmentToolsNotFoundError(
                "Environment missing or invalid (no __class__.__name__)."
            )
        # Tools hold a reference to the environment they mutate (Terrarium >= 0.2.0),
        # so rebuild them whenever the environment instance changes, not just its name.
        if (
            self._env_tools is None
            or self._env_tools_env_name != env_name
            or self._env_tools_env is not environment
        ):
            self._env_tools = instantiate_environment_tools(
                env_name, self.megaboard, environment=environment
            )
            self._env_tools_env_name = env_name
            self._env_tools_env = environment
        return self._env_tools

    def _record_tool_event(
        self,
        *,
        tool_name: str,
        agent_name: str,
        arguments: Dict[str, Any],
        result: Dict[str, Any],
        phase: Optional[str],
        iteration: Optional[int],
    ) -> None:
        self.tool_events.append(
            ToolEvent(
                tool_name=tool_name,
                agent_name=agent_name,
                arguments=dict(arguments or {}),
                result=dict(result or {}),
                phase=phase,
                iteration=iteration,
                planning_round=self.current_planning_round,
            )
        )

    def _scrub_secret_blackboards(self) -> None:
        """
        Remove tool-action trace events from secret blackboards.

        Some environment tool implementations call `blackboard_manager.log_action_to_blackboards(...)`,
        which logs actions to every blackboard an agent belongs to. For secret coalition channels,
        we want them to be used only for communication/scheming, not for action traces.
        """
        try:
            boards = getattr(self.megaboard, "blackboards", None)
            if not isinstance(boards, list):
                return
            for bb in boards:
                tmpl = getattr(bb, "template", None)
                if not isinstance(tmpl, dict):
                    continue
                is_secret = (
                    bool(tmpl.get("secret_channel"))
                    or str(tmpl.get("visibility") or "").lower() == "secret"
                )
                if not is_secret:
                    continue
                logs = getattr(bb, "logs", None)
                if not isinstance(logs, list) or not logs:
                    continue
                bb.logs = [
                    e
                    for e in logs
                    if not (
                        isinstance(e, dict)
                        and str(e.get("kind") or "") == "action_executed"
                    )
                ]
        except Exception:
            return

    async def _prefetch_blackboard_events(
        self,
        agent_name: str,
        *,
        phase: Optional[str],
        iteration: Optional[int],
    ) -> Dict[str, str]:
        blackboard_ids = self.megaboard.get_agent_blackboards(agent_name)

        def _sort_key(bb_id: str) -> int:
            try:
                return int(bb_id)
            except Exception:
                return 0

        contexts: Dict[str, str] = {}
        for bb_id_str in sorted(blackboard_ids, key=_sort_key):
            try:
                bb_id_int = int(bb_id_str)
            except Exception:
                continue

            # Prefetch is automatic; avoid polluting tool_events with synthetic get_blackboard_events calls.
            events = self.megaboard.get(bb_id_int, agent_name, limit=None)
            contexts[bb_id_str] = format_blackboard_events_for_prompt(
                events if isinstance(events, list) else []
            )

        return contexts

    async def environment_handle_tool_call(
        self,
        tool_name: str,
        agent_name: str,
        arguments: Dict[str, Any],
        *,
        phase: Optional[str] = None,
        iteration: Optional[int] = None,
        round_num: Optional[int] = None,
    ) -> Dict[str, Any]:
        policy = self.active_phase_policy(phase)
        if policy is not None and policy.block_environment_tool_calls:
            # The agent still sees the environment tools in its prompt, but any
            # call is dropped here so it cannot probe the environment. The
            # blackboard tools (`post_message`) flow through
            # `blackboard_handle_tool_call` and are unaffected.
            stub_result = {
                "result": None,
                "blocked": True,
                "reason": (
                    f"env tools are blocked during the {phase} phase; "
                    "use `post_message` on the coalition-only blackboard to "
                    "agree on a tool-only protocol"
                ),
            }
            self._record_tool_event(
                tool_name=tool_name,
                agent_name=agent_name,
                arguments=arguments,
                result=stub_result,
                phase=phase,
                iteration=iteration,
            )
            return stub_result

        env = self.environment
        if env is None:
            result = {"error": "Environment not set on protocol."}
            self._record_tool_event(
                tool_name=tool_name,
                agent_name=agent_name,
                arguments=arguments,
                result=result,
                phase=phase,
                iteration=iteration,
            )
            return result

        try:
            env_tools = self._get_env_tools(env)
        except EnvironmentToolsNotFoundError as exc:
            result = {"error": str(exc)}
            self._record_tool_event(
                tool_name=tool_name,
                agent_name=agent_name,
                arguments=arguments,
                result=result,
                phase=phase,
                iteration=iteration,
            )
            return result

        # Terrarium >= 0.2.0 tools mutate the bound environment directly, so there is
        # no env_state snapshot to pass in and no state_updates to apply afterwards.
        # A tool can raise after mutating state; catch it here (with the same result
        # BaseAgent would build) so the call is still recorded and secret boards scrubbed.
        try:
            response = call_env_tool(
                env_tools,
                tool_name,
                agent_name,
                arguments,
                phase=phase,
                iteration=iteration,
                round_num=round_num,
            )
        except Exception as exc:
            logger.exception("Environment tool %s raised", tool_name)
            response = {"error": f"Error executing {tool_name}: {exc}"}

        self._record_tool_event(
            tool_name=tool_name,
            agent_name=agent_name,
            arguments=arguments,
            result=response if isinstance(response, dict) else {"result": response},
            phase=phase,
            iteration=iteration,
        )
        # Ensure secret channels remain communication-only (no action traces).
        self._scrub_secret_blackboards()
        return response

    async def blackboard_handle_tool_call(
        self,
        tool_name: str,
        agent_name: str,
        arguments: Dict[str, Any],
        *,
        phase: Optional[str] = None,
        iteration: Optional[int] = None,
    ) -> Dict[str, Any]:
        policy = self.active_phase_policy(phase)
        if (
            policy is not None
            and policy.blackboard_tools is not None
            and tool_name not in policy.blackboard_tools
        ):
            # The tool is withheld from the agent's tool list for this phase;
            # reject it here as well so a hallucinated call cannot slip through.
            rejection = {
                "result": None,
                "blocked": True,
                "reason": (
                    f"blackboard tool '{tool_name}' is not available during "
                    f"the {phase} phase."
                ),
                "allowed_tools": sorted(policy.blackboard_tools),
            }
            self._record_tool_event(
                tool_name=tool_name,
                agent_name=agent_name,
                arguments=arguments,
                result=rejection,
                phase=phase,
                iteration=iteration,
            )
            return rejection

        # Defense in depth: rewrite any `post_message` call to the coalition-only
        # secret board so the agent cannot accidentally leak the coordination
        # scheme to the public blackboard. We do NOT rewrite arguments if the
        # agent explicitly targeted the secret board already (avoids
        # double-instrumenting).
        if (
            tool_name == "post_message"
            and policy is not None
            and policy.route_post_message_to is not None
        ):
            secret_bb = policy.route_post_message_to
            targeted_bb = arguments.get("blackboard_id")
            if targeted_bb is None:
                arguments = dict(arguments)
                arguments["blackboard_id"] = secret_bb
            else:
                try:
                    if int(targeted_bb) != int(secret_bb):
                        arguments = dict(arguments)
                        arguments["blackboard_id"] = secret_bb
                except (TypeError, ValueError):
                    arguments = dict(arguments)
                    arguments["blackboard_id"] = secret_bb
        # Per-agent `post_message` cap. When the cap is set and the agent is over
        # it, return a structured rejection so the agent can wrap up its
        # coordination message. A limit of 0 rejects all calls.
        if (
            tool_name == "post_message"
            and policy is not None
            and policy.max_post_messages_per_agent is not None
        ):
            limit = policy.max_post_messages_per_agent
            already = self.get_post_message_count(agent_name)
            if already >= limit:
                rejection = {
                    "result": None,
                    "blocked": True,
                    "reason": (
                        f"post_message limit reached for this agent "
                        f"({already}/{limit}); wrap up your coordination."
                    ),
                    "limit": limit,
                    "count": already,
                }
                self._record_tool_event(
                    tool_name=tool_name,
                    agent_name=agent_name,
                    arguments=arguments,
                    result=rejection,
                    phase=phase,
                    iteration=iteration,
                )
                return rejection

        result = self.megaboard.handle_tool_call(
            tool_name,
            agent_name,
            arguments,
            phase=phase,
            iteration=iteration,
        )
        if tool_name == "post_message":
            self._record_post_message_call(agent_name)
        self._record_tool_event(
            tool_name=tool_name,
            agent_name=agent_name,
            arguments=arguments,
            result=result,
            phase=phase,
            iteration=iteration,
        )
        # Be defensive: keep secret channels communication-only.
        self._scrub_secret_blackboards()
        return result

    async def get_all_blackboard_ids(self) -> List[str]:
        return self.megaboard.get_blackboard_string_ids()

    async def get_agent_blackboard_ids(self, agent_name: str) -> List[str]:
        return self.megaboard.get_agent_blackboards(agent_name)

    async def post_system_message(
        self,
        blackboard_id: int,
        kind: str,
        payload: Optional[Dict[str, Any]] = None,
    ) -> str:
        return self.megaboard.post_system_message(blackboard_id, kind, payload)

    async def generate_comm_network(
        self,
        participants: Sequence[str],
        context: str,
        template: Optional[Dict[str, Any]] = None,
    ) -> int:
        blackboard_id = self.megaboard.add_blackboard(list(participants), template)
        self.megaboard.post_system_message(
            blackboard_id,
            kind="context",
            payload={"message": context},
        )
        return blackboard_id

    async def agent_turn(
        self,
        agent: Any,
        agent_name: str,
        agent_context: Any,
        environment: Any,
        iteration: int,
        *,
        phase: str,
        round_num: Optional[int] = None,
        phase_config: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Generic per-phase turn used by terrarium.utils.run_phase_entries.

        Dispatches on the phase: "survey" goes to agent_survey_turn,
        "execution" to agent_execution_turn, and every other phase (planning,
        pre_coordination, ...) to agent_planning_turn with the round number and
        the phase label. If the turn raises, the exception is returned as the
        turn result so run_phase_entries keeps walking and the caller's on_turn
        hook can record an error entry (mirrors the runners' per-turn try/except).
        """
        try:
            if phase == "survey":
                return await self.agent_survey_turn(
                    agent, agent_name, agent_context, environment, iteration
                )
            if phase == "execution":
                return await self.agent_execution_turn(
                    agent,
                    agent_name,
                    agent_context,
                    environment,
                    iteration,
                    round_num=round_num,
                )
            return await self.agent_planning_turn(
                agent,
                agent_name,
                agent_context,
                environment,
                iteration,
                planning_round=int(round_num) if round_num is not None else 1,
                phase=phase,
            )
        except Exception as exc:
            return exc

    async def agent_planning_turn(
        self,
        agent: Any,
        agent_name: str,
        agent_context: Any,
        environment: Any,
        iteration: int,
        planning_round: int,
        phase: str = "planning",
    ) -> Dict[str, Any]:
        """Run one planning-shaped turn for ``agent``.

        ``phase`` labels the turn for the trajectory/tool logs and defaults to
        ``"planning"`` so ordinary iteration planning rounds are unaffected.
        Callers that reuse the planning prompt for another round (e.g. an
        adversary pre-coordination round) pass a distinct phase so it is not
        conflated with a normal planning round.
        """
        self.environment = environment
        self.current_planning_round = planning_round
        blackboard_contexts = await self._prefetch_blackboard_events(
            agent_name,
            phase=phase,
            iteration=iteration,
        )
        prompts = environment.prompts
        return await agent.generate_response(
            agent_name=agent_name,
            agent_context=agent_context,
            blackboard_context=blackboard_contexts,
            communication_protocol=self,
            prompts=prompts,
            phase=phase,
            iteration=iteration,
            round_num=planning_round,
        )

    async def agent_execution_turn(
        self,
        agent: Any,
        agent_name: str,
        agent_context: Any,
        environment: Any,
        iteration: int,
        round_num: Optional[int] = None,
    ) -> Dict[str, Any]:
        self.environment = environment
        self.current_planning_round = None
        blackboard_contexts = await self._prefetch_blackboard_events(
            agent_name,
            phase="execution",
            iteration=iteration,
        )
        prompts = environment.prompts
        return await agent.generate_response(
            agent_name=agent_name,
            agent_context=agent_context,
            blackboard_context=blackboard_contexts,
            communication_protocol=self,
            prompts=prompts,
            phase="execution",
            iteration=iteration,
            round_num=round_num or 0,
        )

    async def agent_survey_turn(
        self,
        agent: Any,
        agent_name: str,
        agent_context: Any,
        environment: Any,
        iteration: int,
    ) -> Dict[str, Any]:
        """Optional post-run survey turn to elicit beliefs without posting to blackboards."""
        self.environment = environment
        self.current_planning_round = None
        blackboard_contexts = await self._prefetch_blackboard_events(
            agent_name,
            phase="survey",
            iteration=iteration,
        )
        prompts = environment.prompts
        return await agent.generate_response(
            agent_name=agent_name,
            agent_context=agent_context,
            blackboard_context=blackboard_contexts,
            communication_protocol=self,
            prompts=prompts,
            phase="survey",
            iteration=iteration,
            round_num=0,
        )
