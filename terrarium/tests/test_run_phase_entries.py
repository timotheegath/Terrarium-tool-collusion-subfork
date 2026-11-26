import unittest
from typing import Any, Dict, List, Optional

from terrarium.utils import run_phase_entries


class _Agent:
    def __init__(self, name: str) -> None:
        self.name = name


class _Environment:
    def __init__(self, agents: List[_Agent]) -> None:
        self.agents = agents
        self.context_calls: List[Dict[str, Any]] = []

    def build_agent_context(
        self, agent_name: str, phase: str, iteration: int, **kwargs: Any
    ) -> Dict[str, Any]:
        call = {"agent_name": agent_name, "phase": phase, "iteration": iteration, **kwargs}
        self.context_calls.append(call)
        return dict(call)


class _SequentialProtocol:
    """Mimics the base protocol: only planning/execution turn methods."""

    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    async def agent_planning_turn(
        self, agent, agent_name, agent_context, environment, iteration, planning_round
    ):
        self.calls.append(
            {
                "method": "planning",
                "agent_name": agent_name,
                "iteration": iteration,
                "planning_round": planning_round,
                "agent_context": agent_context,
            }
        )
        return {"phase": "planning", "agent_name": agent_name}

    async def agent_execution_turn(
        self, agent, agent_name, agent_context, environment, iteration
    ):
        self.calls.append(
            {
                "method": "execution",
                "agent_name": agent_name,
                "iteration": iteration,
                "agent_context": agent_context,
            }
        )
        return {"phase": "execution", "agent_name": agent_name}


class _GenericProtocol(_SequentialProtocol):
    """Mimics a protocol with a generic agent_turn for every phase."""

    async def agent_turn(
        self,
        agent,
        agent_name,
        agent_context,
        environment,
        iteration,
        phase,
        round_num: Optional[int] = None,
        phase_config: Optional[Dict[str, Any]] = None,
    ):
        self.calls.append(
            {
                "method": "generic",
                "agent_name": agent_name,
                "phase": phase,
                "iteration": iteration,
                "round_num": round_num,
                "phase_config": phase_config,
                "agent_context": agent_context,
            }
        )
        return {"phase": phase, "agent_name": agent_name}


def _entries() -> List[Dict[str, Any]]:
    return [
        {"phase": "planning", "rounds": 2},
        {"phase": "execution", "rounds": 1},
    ]


class RunPhaseEntriesTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.agents = [_Agent("agent_0"), _Agent("agent_1")]
        self.environment = _Environment(self.agents)
        self.protocol = _SequentialProtocol()

    async def test_walks_rounds_and_agents_in_order(self):
        turns = await run_phase_entries(
            _entries(),
            environment=self.environment,
            communication_protocol=self.protocol,
            iteration=3,
            progress=False,
        )

        self.assertEqual(
            [(c["method"], c["agent_name"], c.get("planning_round")) for c in self.protocol.calls],
            [
                ("planning", "agent_0", 1),
                ("planning", "agent_1", 1),
                ("planning", "agent_0", 2),
                ("planning", "agent_1", 2),
                ("execution", "agent_0", None),
                ("execution", "agent_1", None),
            ],
        )
        self.assertEqual(len(turns), 6)
        self.assertEqual(
            [(t["phase"], t["agent_name"], t["round"], t["iteration"]) for t in turns],
            [
                ("planning", "agent_0", 1, 3),
                ("planning", "agent_1", 1, 3),
                ("planning", "agent_0", 2, 3),
                ("planning", "agent_1", 2, 3),
                ("execution", "agent_0", 1, 3),
                ("execution", "agent_1", 1, 3),
            ],
        )
        for turn in turns:
            self.assertIs(turn["agent"], self.agents[0] if turn["agent_name"] == "agent_0" else self.agents[1])
            self.assertIn("result", turn)

    async def test_planning_context_carries_planning_round_execution_does_not(self):
        await run_phase_entries(
            _entries(),
            environment=self.environment,
            communication_protocol=self.protocol,
            iteration=1,
            progress=False,
        )

        planning_calls = [c for c in self.environment.context_calls if c["phase"] == "planning"]
        execution_calls = [c for c in self.environment.context_calls if c["phase"] == "execution"]
        self.assertEqual([c["planning_round"] for c in planning_calls], [1, 1, 2, 2])
        self.assertTrue(all("planning_round" not in c for c in execution_calls))
        self.assertTrue(all(c["iteration"] == 1 for c in self.environment.context_calls))

    async def test_entry_name_key_and_default_rounds(self):
        turns = await run_phase_entries(
            [{"name": "execution"}],
            environment=self.environment,
            communication_protocol=self.protocol,
            iteration=1,
            progress=False,
        )

        self.assertEqual([c["method"] for c in self.protocol.calls], ["execution", "execution"])
        self.assertEqual([t["round"] for t in turns], [1, 1])

    async def test_agents_override(self):
        turns = await run_phase_entries(
            [{"phase": "execution"}],
            environment=self.environment,
            communication_protocol=self.protocol,
            iteration=1,
            agents=[self.agents[1]],
            progress=False,
        )

        self.assertEqual([t["agent_name"] for t in turns], ["agent_1"])

    async def test_on_turn_hook_receives_turn_with_result(self):
        recorded: List[Dict[str, Any]] = []

        def on_turn(turn: Dict[str, Any]) -> None:
            recorded.append(dict(turn))

        turns = await run_phase_entries(
            [{"phase": "execution"}],
            environment=self.environment,
            communication_protocol=self.protocol,
            iteration=2,
            on_turn=on_turn,
            progress=False,
        )

        self.assertEqual(len(recorded), 2)
        self.assertEqual(recorded[0]["result"], {"phase": "execution", "agent_name": "agent_0"})
        self.assertEqual([t["round"] for t in turns], [1, 1])

    async def test_async_on_turn_hook_is_awaited(self):
        recorded: List[str] = []

        async def on_turn(turn: Dict[str, Any]) -> None:
            recorded.append(turn["agent_name"])

        await run_phase_entries(
            [{"phase": "execution"}],
            environment=self.environment,
            communication_protocol=self.protocol,
            iteration=1,
            on_turn=on_turn,
            progress=False,
        )

        self.assertEqual(recorded, ["agent_0", "agent_1"])

    async def test_build_turn_can_replace_turn(self):
        def build_turn(turn: Dict[str, Any]) -> Dict[str, Any]:
            turn["agent_context"]["injected"] = True
            return turn

        await run_phase_entries(
            [{"phase": "execution"}],
            environment=self.environment,
            communication_protocol=self.protocol,
            iteration=1,
            build_turn=build_turn,
            progress=False,
        )

        self.assertTrue(all(c["agent_context"].get("injected") for c in self.protocol.calls))

    async def test_async_build_turn_returning_none_keeps_turn(self):
        async def build_turn(turn: Dict[str, Any]) -> None:
            turn["agent_context"]["touched"] = True

        turns = await run_phase_entries(
            [{"phase": "execution"}],
            environment=self.environment,
            communication_protocol=self.protocol,
            iteration=1,
            build_turn=build_turn,
            progress=False,
        )

        self.assertTrue(all(t["agent_context"].get("touched") for t in turns))

    async def test_generic_agent_turn_protocol_receives_every_phase(self):
        protocol = _GenericProtocol()

        turns = await run_phase_entries(
            _entries(),
            environment=self.environment,
            communication_protocol=protocol,
            iteration=1,
            progress=False,
        )

        self.assertEqual(
            [(c["method"], c["phase"], c["round_num"]) for c in protocol.calls],
            [
                ("generic", "planning", 1),
                ("generic", "planning", 1),
                ("generic", "planning", 2),
                ("generic", "planning", 2),
                ("generic", "execution", 1),
                ("generic", "execution", 1),
            ],
        )
        self.assertTrue(all(c["phase_config"] is not None for c in protocol.calls))
        self.assertTrue(
            all("planning_round" in c["agent_context"] for c in protocol.calls)
        )
        self.assertEqual(len(turns), 6)

    async def test_empty_entries_runs_no_turns(self):
        turns = await run_phase_entries(
            [],
            environment=self.environment,
            communication_protocol=self.protocol,
            iteration=1,
            progress=False,
        )

        self.assertEqual(turns, [])
        self.assertEqual(self.protocol.calls, [])

    async def test_malformed_entries_raise_value_error(self):
        base = dict(
            environment=self.environment,
            communication_protocol=self.protocol,
            iteration=1,
            progress=False,
        )
        for bad_entries in (
            ["planning"],
            [{"rounds": 1}],
            [{"phase": "  "}],
            [{"phase": "planning", "rounds": 0}],
            [{"phase": "planning", "rounds": True}],
            [{"phase": "planning", "rounds": "2"}],
        ):
            with self.subTest(entries=bad_entries):
                with self.assertRaises(ValueError):
                    await run_phase_entries(bad_entries, **base)


if __name__ == "__main__":
    unittest.main()
