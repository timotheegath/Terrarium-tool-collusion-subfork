"""Tests for the config-driven phase sequence (simulation.phases).

No LLM access: resolve_phases and the toolset fallbacks are pure functions,
and SequentialCommunicationProtocol.agent_turn is driven with a stub agent
that records the kwargs it is called with.
"""

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

from terrarium.communication_protocols.sequential import SequentialCommunicationProtocol
from terrarium.core.blackboard import Megaboard
from terrarium.tools.discovery import ToolsetDiscovery
from terrarium.tools.prompts import get_phase_tool_instructions
from terrarium.utils import (
    load_phase_prompt_assets,
    resolve_phases,
    validate_config,
)


def _phase(name, rounds, blocked_tools=None, prompt_file=None, prompt_hook=None):
    """Expected resolved phase dict (all optional keys present)."""
    return {
        "name": name,
        "rounds": rounds,
        "blocked_tools": blocked_tools or [],
        "prompt_file": prompt_file,
        "prompt_hook": prompt_hook,
    }


class ResolvePhasesTests(unittest.TestCase):
    def test_default_derives_from_max_planning_rounds(self):
        self.assertEqual(
            resolve_phases({"max_planning_rounds": 3}),
            [
                _phase("planning", 3),
                _phase("execution", 1),
            ],
        )

    def test_default_when_max_planning_rounds_missing_or_falsy(self):
        expected = [
            _phase("planning", 1),
            _phase("execution", 1),
        ]
        for simulation_config in (
            None,
            {},
            {"max_planning_rounds": None},
            {"max_planning_rounds": 0},
        ):
            with self.subTest(simulation_config=simulation_config):
                self.assertEqual(resolve_phases(simulation_config), expected)

    def test_explicit_phases_pass_through_in_order(self):
        phases = [
            {"name": "planning", "rounds": 2},
            {"name": "critique", "rounds": 1},
            {"name": "execution", "rounds": 1},
        ]
        self.assertEqual(
            resolve_phases({"phases": phases}),
            [_phase("planning", 2), _phase("critique", 1), _phase("execution", 1)],
        )

    def test_explicit_phases_normalize_name_and_rounds(self):
        self.assertEqual(
            resolve_phases({"phases": [{"name": " review ", "rounds": "2"}]}),
            [_phase("review", 2)],
        )

    def test_invalid_phase_entries_raise_value_error(self):
        bad_configs = [
            {"phases": "planning"},
            {"phases": {}},
            {"phases": []},
            {"phases": ["planning"]},
            {"phases": [{"rounds": 1}]},
            {"phases": [{"name": "", "rounds": 1}]},
            {"phases": [{"name": "   ", "rounds": 1}]},
            {"phases": [{"name": 7, "rounds": 1}]},
            {"phases": [{"name": "x"}]},
            {"phases": [{"name": "x", "rounds": None}]},
            {"phases": [{"name": "x", "rounds": 0}]},
            {"phases": [{"name": "x", "rounds": -2}]},
            {"phases": [{"name": "x", "rounds": 1.5}]},
            {"phases": [{"name": "x", "rounds": True}]},
            {"phases": [{"name": "x", "rounds": "many"}]},
            # per-phase optional keys
            {"phases": [{"name": "x", "rounds": 1, "blocked_tools": "*", "extra": 1}]},
            {"phases": [{"name": "x", "rounds": 1, "blocked_tools": [1]}]},
            {"phases": [{"name": "x", "rounds": 1, "blocked_tools": ["  "]}]},
            {"phases": [{"name": "x", "rounds": 1, "blocked_tools": 7}]},
            {"phases": [{"name": "x", "rounds": 1, "prompt_file": ""}]},
            {"phases": [{"name": "x", "rounds": 1, "prompt_file": 42}]},
            {"phases": [{"name": "x", "rounds": 1, "prompt_hook": "no_colon"}]},
            {"phases": [{"name": "x", "rounds": 1, "prompt_hook": ":func"}]},
            {"phases": [{"name": "x", "rounds": 1, "prompt_hook": "mod:"}]},
            {"phases": [{"name": "x", "rounds": 1, "prompt_hook": 7}]},
            # unknown keys are rejected so typos fail fast
            {"phases": [{"name": "x", "rounds": 1, "block_tools": ["*"]}]},
        ]
        for simulation_config in bad_configs:
            with self.subTest(simulation_config=simulation_config), self.assertRaises(ValueError):
                resolve_phases(simulation_config)

    def test_per_phase_keys_pass_through_and_normalize(self):
        resolved = resolve_phases(
            {
                "phases": [
                    {
                        "name": " critique ",
                        "rounds": 1,
                        "blocked_tools": ["*", " post_message "],
                        "prompt_file": " prompts/critique.txt ",
                        "prompt_hook": "my_mod:my_hook",
                    }
                ]
            }
        )
        self.assertEqual(
            resolved,
            [
                _phase(
                    "critique",
                    1,
                    blocked_tools=["*", "post_message"],
                    prompt_file="prompts/critique.txt",
                    prompt_hook="my_mod:my_hook",
                )
            ],
        )

    def test_single_string_blocked_tools_coerced_to_list(self):
        resolved = resolve_phases(
            {"phases": [{"name": "x", "rounds": 1, "blocked_tools": "*"}]}
        )
        self.assertEqual(resolved[0]["blocked_tools"], ["*"])

    def test_non_dict_simulation_config_raises(self):
        with self.assertRaises(ValueError):
            resolve_phases(["planning"])  # type: ignore[arg-type]

    def test_phases_make_max_planning_rounds_optional_in_validate_config(self):
        config = {
            "simulation": {
                "max_iterations": 1,
                "seed": 1,
                "phases": [
                    {"name": "planning", "rounds": 1},
                    {"name": "execution", "rounds": 1},
                ],
            },
            "environment": {"name": "MeetingSchedulingEnvironment"},
            "communication_network": {"topology": "complete", "num_agents": 2},
            "llm": {"provider": "openai"},
        }
        validate_config(config)  # must not raise
        with self.assertRaises(ValueError):
            validate_config(
                {**config, "simulation": {"max_iterations": 1, "seed": 1, "phases": []}}
            )


class BlackboardToolsetFallbackTests(unittest.TestCase):
    def setUp(self):
        self.discovery = ToolsetDiscovery()

    def _names(self, phase: str, **kwargs: Any) -> List[str]:
        return [
            tool["function"]["name"]
            for tool in self.discovery.get_tools_for_blackboard(phase, **kwargs)
        ]

    def test_planning_gets_blackboard_tools(self):
        self.assertIn("post_message", self._names("planning"))

    def test_execution_gets_no_blackboard_tools(self):
        self.assertEqual(self.discovery.get_tools_for_blackboard("execution"), [])

    def test_unknown_phase_gets_the_planning_toolset(self):
        unknown = self.discovery.get_tools_for_blackboard("critique")
        planning = self.discovery.get_tools_for_blackboard("planning")
        self.assertEqual(unknown, planning)
        self.assertIn("post_message", [t["function"]["name"] for t in unknown])

    def test_unknown_phase_respects_retrieval_and_private_channel_flags(self):
        names = self._names("critique", retrieval_enabled=True)
        self.assertIn("post_message", names)
        self.assertIn("recall", names)
        channel_env = self._names(
            "critique", environment_name="IsThisSeatTakenEnvironment"
        )
        self.assertIn("create_channel", channel_env)


_INSTRUCTION_MAP = {"planning": "PLAN-BLOCK", "execution": "EXEC-BLOCK"}


class PhaseToolInstructionFallbackTests(unittest.TestCase):
    def test_known_phases_are_case_insensitive(self):
        self.assertEqual(
            get_phase_tool_instructions(_INSTRUCTION_MAP, "planning"), "PLAN-BLOCK"
        )
        self.assertEqual(
            get_phase_tool_instructions(_INSTRUCTION_MAP, "Execution"), "EXEC-BLOCK"
        )

    def test_unknown_phase_falls_back_to_execution_block(self):
        for phase in ("critique", "review", None, ""):
            with self.subTest(phase=phase):
                self.assertEqual(
                    get_phase_tool_instructions(_INSTRUCTION_MAP, phase),
                    "EXEC-BLOCK",
                )

    def test_missing_execution_key_returns_empty_string(self):
        self.assertEqual(
            get_phase_tool_instructions({"planning": "PLAN-BLOCK"}, "critique"), ""
        )

    def test_empty_or_missing_map_returns_empty_string(self):
        self.assertEqual(get_phase_tool_instructions({}, "planning"), "")
        self.assertEqual(get_phase_tool_instructions(None, "planning"), "")


class _RecordingAgent:
    """Stub agent: records the kwargs generate_response is called with."""

    def __init__(self):
        self.client = object()
        self.model_name = "stub-model"
        self.calls: List[Dict[str, Any]] = []

    async def generate_response(self, **kwargs: Any) -> Dict[str, Any]:
        self.calls.append(kwargs)
        return {"response": "ok"}


class _StubEnvironment:
    def __init__(self):
        self.prompts = object()


def _make_protocol() -> SequentialCommunicationProtocol:
    """Build a protocol without running __init__ (avoids logger filesystem setup)."""
    protocol = SequentialCommunicationProtocol.__new__(SequentialCommunicationProtocol)
    protocol.config = {"simulation": {}, "llm": {}}
    protocol.simulation_config = protocol.config["simulation"]
    protocol.tool_logger = None
    protocol.run_timestamp = None
    protocol.blackboard_logger = None
    protocol.compaction_enabled = False
    protocol.compaction_logger = None
    protocol._compaction_caches = {}
    protocol.megaboard = Megaboard()
    protocol.environment = None
    protocol.environment_tools = None
    protocol._environment_tools_name = None
    return protocol


class AgentTurnTests(unittest.IsolatedAsyncioTestCase):
    def _bind(self, protocol: SequentialCommunicationProtocol, environment) -> None:
        protocol.environment = environment
        # Pretend environment tools are already initialized for this env class.
        protocol.environment_tools = object()
        protocol._environment_tools_name = type(environment).__name__

    def _spy(self, protocol: SequentialCommunicationProtocol):
        prefetch_calls: List[Dict[str, Any]] = []
        log_calls: List[Dict[str, Any]] = []
        original_prefetch = protocol._prefetch_blackboard_events
        original_log = protocol._log_blackboard_states

        async def prefetch_spy(agent_name, **kwargs):
            prefetch_calls.append({"agent_name": agent_name, **kwargs})
            return await original_prefetch(agent_name, **kwargs)

        def log_spy(**kwargs):
            log_calls.append(kwargs)
            original_log(**kwargs)

        protocol._prefetch_blackboard_events = prefetch_spy
        protocol._log_blackboard_states = log_spy
        return prefetch_calls, log_calls

    async def test_agent_turn_forwards_phase_and_round_num(self):
        protocol = _make_protocol()
        environment = _StubEnvironment()
        self._bind(protocol, environment)
        prefetch_calls, log_calls = self._spy(protocol)
        agent = _RecordingAgent()

        await protocol.agent_turn(
            agent, "agent_0", {"ctx": 1}, environment, 3, phase="review", round_num=2
        )

        self.assertEqual(len(agent.calls), 1)
        call = agent.calls[0]
        self.assertEqual(call["agent_name"], "agent_0")
        self.assertEqual(call["agent_context"], {"ctx": 1})
        self.assertEqual(call["phase"], "review")
        self.assertEqual(call["round_num"], 2)
        self.assertEqual(call["iteration"], 3)
        self.assertIs(call["communication_protocol"], protocol)
        self.assertIs(call["prompts"], environment.prompts)
        # phase is passed through to the blackboard prefetch and state logging
        self.assertEqual(prefetch_calls[0]["phase"], "review")
        self.assertEqual(prefetch_calls[0]["iteration"], 3)
        self.assertEqual(log_calls[0]["phase"], "review")
        self.assertEqual(log_calls[0]["agent_name"], "agent_0")
        self.assertEqual(log_calls[0]["planning_round"], 2)

    async def test_agent_turn_without_round_num_keeps_agent_default(self):
        protocol = _make_protocol()
        environment = _StubEnvironment()
        self._bind(protocol, environment)
        prefetch_calls, log_calls = self._spy(protocol)
        agent = _RecordingAgent()

        await protocol.agent_turn(agent, "agent_0", {}, environment, 1, phase="execution")

        call = agent.calls[0]
        self.assertEqual(call["phase"], "execution")
        # round_num is not forwarded, so generate_response keeps its default (0)
        self.assertNotIn("round_num", call)
        self.assertEqual(prefetch_calls[0]["phase"], "execution")
        self.assertIsNone(log_calls[0].get("planning_round"))

    async def test_legacy_wrappers_delegate_to_agent_turn_with_fixed_phase(self):
        protocol = _make_protocol()
        environment = _StubEnvironment()
        agent = _RecordingAgent()
        seen: List[Any] = []

        async def fake_turn(*args, **kwargs):
            seen.append((args, kwargs))

        protocol.agent_turn = fake_turn  # type: ignore[method-assign]

        await protocol.agent_planning_turn(agent, "agent_0", {"ctx": 2}, environment, 5, 3)
        await protocol.agent_execution_turn(agent, "agent_0", {"ctx": 2}, environment, 5)

        self.assertEqual(len(seen), 2)
        planning_args, planning_kwargs = seen[0]
        self.assertEqual(
            planning_args, (agent, "agent_0", {"ctx": 2}, environment, 5)
        )
        self.assertEqual(planning_kwargs, {"phase": "planning", "round_num": 3})
        execution_args, execution_kwargs = seen[1]
        self.assertEqual(
            execution_args, (agent, "agent_0", {"ctx": 2}, environment, 5)
        )
        self.assertEqual(execution_kwargs, {"phase": "execution", "round_num": None})

    async def test_planning_and_execution_wrappers_end_to_end(self):
        protocol = _make_protocol()
        environment = _StubEnvironment()
        self._bind(protocol, environment)
        agent = _RecordingAgent()

        await protocol.agent_planning_turn(agent, "agent_0", {}, environment, 2, 4)
        await protocol.agent_execution_turn(agent, "agent_0", {}, environment, 2)

        planning_call, execution_call = agent.calls
        self.assertEqual(planning_call["phase"], "planning")
        self.assertEqual(planning_call["round_num"], 4)
        self.assertEqual(planning_call["iteration"], 2)
        self.assertEqual(execution_call["phase"], "execution")
        self.assertNotIn("round_num", execution_call)
        self.assertEqual(execution_call["iteration"], 2)

    async def test_agent_turn_forwards_phase_config(self):
        protocol = _make_protocol()
        environment = _StubEnvironment()
        self._bind(protocol, environment)
        agent = _RecordingAgent()
        phase_config = {"name": "critique", "rounds": 1, "prompt_text": "be nice"}

        await protocol.agent_turn(
            agent, "agent_0", {}, environment, 1, phase="critique", round_num=1,
            phase_config=phase_config,
        )

        self.assertIs(agent.calls[0]["phase_config"], phase_config)

    async def test_agent_turn_omits_phase_config_when_not_given(self):
        protocol = _make_protocol()
        environment = _StubEnvironment()
        self._bind(protocol, environment)
        agent = _RecordingAgent()

        await protocol.agent_turn(agent, "agent_0", {}, environment, 1, phase="planning", round_num=1)

        self.assertNotIn("phase_config", agent.calls[0])


class LoadPhasePromptAssetsTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp_dir = Path(self._tmp.name)
        # A config "file" so _config_path has a directory to resolve against.
        self.config_path = self.tmp_dir / "run.yaml"
        self.config_path.write_text("simulation: {}\n", encoding="utf-8")
        # A hook module importable from the config directory.
        (self.tmp_dir / "phase_hooks.py").write_text(
            "def critique_hook(agent_name, **kwargs):\n"
            "    return f'hook says hi to {agent_name}'\n",
            encoding="utf-8",
        )
        sys.path.insert(0, str(self.tmp_dir))
        self.addCleanup(lambda: sys.path.remove(str(self.tmp_dir)))
        self.addCleanup(lambda: sys.modules.pop("phase_hooks", None))

    def _config(self, phases):
        return {"_config_path": str(self.config_path), "simulation": {"phases": phases}}

    def test_prompt_file_resolved_relative_to_config_dir(self):
        (self.tmp_dir / "prompts").mkdir()
        (self.tmp_dir / "prompts" / "critique.txt").write_text(
            "CRITIQUE PROMPT", encoding="utf-8"
        )
        phases = load_phase_prompt_assets(
            self._config(
                [{"name": "critique", "rounds": 1, "prompt_file": "prompts/critique.txt"}]
            )
        )
        self.assertEqual(phases[0]["prompt_text"], "CRITIQUE PROMPT")
        self.assertIsNone(phases[0]["prompt_hook"])

    def test_prompt_file_absolute_path(self):
        prompt_path = self.tmp_dir / "abs.txt"
        prompt_path.write_text("ABS PROMPT", encoding="utf-8")
        phases = load_phase_prompt_assets(
            self._config(
                [{"name": "critique", "rounds": 1, "prompt_file": str(prompt_path)}]
            )
        )
        self.assertEqual(phases[0]["prompt_text"], "ABS PROMPT")

    def test_missing_prompt_file_raises_value_error(self):
        with self.assertRaises(ValueError):
            load_phase_prompt_assets(
                self._config([{"name": "critique", "rounds": 1, "prompt_file": "nope.txt"}])
            )

    def test_prompt_hook_imported(self):
        phases = load_phase_prompt_assets(
            self._config(
                [{"name": "critique", "rounds": 1, "prompt_hook": "phase_hooks:critique_hook"}]
            )
        )
        hook = phases[0]["prompt_hook"]
        self.assertTrue(callable(hook))
        self.assertEqual(hook("agent_0"), "hook says hi to agent_0")
        self.assertEqual(phases[0]["prompt_text"], "")

    def test_bad_prompt_hook_raises_value_error(self):
        with self.assertRaises(ValueError):
            load_phase_prompt_assets(
                self._config(
                    [{"name": "x", "rounds": 1, "prompt_hook": "no_such_mod_xyz:hook"}]
                )
            )
        with self.assertRaises(ValueError):
            load_phase_prompt_assets(
                self._config([{"name": "x", "rounds": 1, "prompt_hook": "phase_hooks:missing"}])
            )

    def test_phases_without_prompt_keys_get_empty_assets(self):
        phases = load_phase_prompt_assets(
            self._config([{"name": "planning", "rounds": 2}, {"name": "execution", "rounds": 1}])
        )
        for phase in phases:
            self.assertEqual(phase["prompt_text"], "")
            self.assertIsNone(phase["prompt_hook"])

    def test_default_phases_get_empty_assets(self):
        config = {"_config_path": str(self.config_path), "simulation": {"max_planning_rounds": 2}}
        phases = load_phase_prompt_assets(config)
        self.assertEqual([p["name"] for p in phases], ["planning", "execution"])
        for phase in phases:
            self.assertEqual(phase["prompt_text"], "")
            self.assertIsNone(phase["prompt_hook"])


class _StubClient:
    async def get_external_tool_names(self):
        return set()


class AgentPhasePromptAndGatingTests(unittest.IsolatedAsyncioTestCase):
    """BaseAgent per-phase prompt injection and tool gating, no LLM access."""

    def _agent(self, simulation_config):
        from terrarium.agents.base import BaseAgent

        agent = BaseAgent.__new__(BaseAgent)
        agent.communication_protocol = SimpleNamespace(
            simulation_config=simulation_config, environment=None
        )
        agent.current_agent_name = "agent_0"
        agent.current_phase = "critique"
        agent.current_iteration = 1
        agent.current_round = 1
        agent.current_phase_config = None
        agent.tool_logger = None
        agent.trajectory_logger = None
        agent.toolset_discovery = ToolsetDiscovery()
        agent.environment_name = "HospitalEnvironment"
        agent.retrieval_enabled = False
        agent.client = _StubClient()
        return agent

    def test_phase_prompt_block_combines_file_text_and_hook(self):
        agent = self._agent({})
        agent.current_phase_config = {
            "prompt_text": "FILE TEXT",
            "prompt_hook": lambda agent_name, **kw: f"HOOK:{agent_name}",
        }
        block = agent._build_phase_prompt_block("agent_0", {}, {})
        self.assertIn("PHASE INSTRUCTIONS (critique)", block)
        self.assertIn("FILE TEXT", block)
        self.assertIn("HOOK:agent_0", block)

    def test_phase_prompt_block_empty_without_phase_config(self):
        agent = self._agent({})
        self.assertEqual(agent._build_phase_prompt_block("agent_0", {}, {}), "")

    def test_phase_prompt_hook_signature_filtering(self):
        agent = self._agent({})
        agent.current_phase_config = {
            "prompt_hook": lambda phase: f"only-phase:{phase}",
        }
        block = agent._build_phase_prompt_block("agent_0", {}, {})
        self.assertIn("only-phase:critique", block)

    async def test_blocked_tool_call_returns_blocked_response(self):
        agent = self._agent(
            {"phases": [{"name": "critique", "rounds": 1, "blocked_tools": ["schedule_patient"]}]}
        )
        result = await agent._execute_tool_call("schedule_patient", {"patient_id": "P1"})
        self.assertEqual(result["blocked"], True)
        self.assertEqual(result["status"], "blocked")
        self.assertIn("critique", result["error"])

    async def test_unblocked_tool_is_not_short_circuited(self):
        agent = self._agent(
            {"phases": [{"name": "critique", "rounds": 1, "blocked_tools": ["*"]}]}
        )
        # post_message is a blackboard tool name, so "*" must not block it; the
        # call should proceed past the gate and fail later in dispatch instead.
        result = await agent._execute_tool_call("post_message", {"message": "hi"})
        self.assertNotEqual(result.get("status"), "blocked")


if __name__ == "__main__":
    unittest.main()
