"""Tests for phase-entry `when`/`agents`/`max_calls` and per-agent tool gating."""

import unittest
from types import SimpleNamespace

from terrarium.communication_protocols.sequential import SequentialCommunicationProtocol
from terrarium.core.blackboard import Megaboard
from terrarium.tools.discovery import ToolsetDiscovery
from terrarium.tools.gates import (
    call_limit_response,
    secret_channel_rewrite,
    tool_blocked,
)
from terrarium.utils import (
    phase_max_calls,
    resolve_phase_agents,
    resolve_phases,
    split_phases_by_when,
)


class ResolvePhasesExtensionsTests(unittest.TestCase):
    def test_when_agents_max_calls_pass_through(self):
        sim = {
            "phases": [
                {
                    "name": "pre_coordination",
                    "rounds": 2,
                    "when": "before",
                    "agents": ["A", "B"],
                    "max_calls": {"post_message": 1},
                },
                {"name": "planning", "rounds": 1},
                {"name": "survey", "rounds": 1, "when": "after"},
            ]
        }
        phases = resolve_phases(sim)
        self.assertEqual(
            phases[0],
            {
                "name": "pre_coordination",
                "rounds": 2,
                "when": "before",
                "agents": ["A", "B"],
                "max_calls": {"post_message": 1},
            },
        )
        self.assertEqual(phases[1]["when"], "each")
        self.assertNotIn("agents", phases[1])
        self.assertEqual(phases[2]["when"], "after")

    def test_indices_form(self):
        phases = resolve_phases(
            {"phases": [{"name": "p", "rounds": 1, "agents": {"indices": [0, 2]}}]}
        )
        self.assertEqual(phases[0]["agents"], {"indices": [0, 2]})
        self.assertEqual(resolve_phase_agents(phases[0], ["A", "B", "C"]), ["A", "C"])

    def test_invalid_when_rejected(self):
        with self.assertRaises(ValueError):
            resolve_phases({"phases": [{"name": "p", "rounds": 1, "when": "sideways"}]})

    def test_invalid_agents_rejected(self):
        with self.assertRaises(ValueError):
            resolve_phases({"phases": [{"name": "p", "rounds": 1, "agents": []}]})
        with self.assertRaises(ValueError):
            resolve_phases(
                {"phases": [{"name": "p", "rounds": 1, "agents": {"indices": [-1]}}]}
            )

    def test_invalid_max_calls_rejected(self):
        with self.assertRaises(ValueError):
            resolve_phases(
                {"phases": [{"name": "p", "rounds": 1, "max_calls": {"post_message": 0}}]}
            )
        with self.assertRaises(ValueError):
            resolve_phases({"phases": [{"name": "p", "rounds": 1, "max_calls": []}]})

    def test_legacy_fallback_marks_each(self):
        phases = resolve_phases({"max_planning_rounds": 2})
        self.assertEqual(
            phases,
            [
                {"name": "planning", "rounds": 2, "when": "each"},
                {"name": "execution", "rounds": 1, "when": "each"},
            ],
        )


class PhaseAgentResolutionTests(unittest.TestCase):
    def test_no_agents_key_means_all(self):
        self.assertIsNone(resolve_phase_agents({"name": "p"}, ["A", "B"]))

    def test_name_subset_preserves_env_order(self):
        phase = {"name": "p", "agents": ["B", "A"]}
        self.assertEqual(resolve_phase_agents(phase, ["A", "B", "C"]), ["A", "B"])

    def test_unknown_agent_raises(self):
        with self.assertRaises(ValueError):
            resolve_phase_agents({"name": "p", "agents": ["Z"]}, ["A", "B"])

    def test_index_out_of_range_raises(self):
        with self.assertRaises(ValueError):
            resolve_phase_agents({"name": "p", "agents": {"indices": [5]}}, ["A", "B"])

    def test_split_phases_by_when(self):
        phases = resolve_phases(
            {
                "phases": [
                    {"name": "pre", "rounds": 1, "when": "before"},
                    {"name": "planning", "rounds": 1},
                    {"name": "execution", "rounds": 1},
                    {"name": "survey", "rounds": 1, "when": "after"},
                ]
            }
        )
        groups = split_phases_by_when(phases)
        self.assertEqual([p["name"] for p in groups["before"]], ["pre"])
        self.assertEqual([p["name"] for p in groups["each"]], ["planning", "execution"])
        self.assertEqual([p["name"] for p in groups["after"]], ["survey"])

    def test_phase_max_calls_lookup(self):
        phases = resolve_phases(
            {"phases": [{"name": "pre", "rounds": 1, "max_calls": {"post_message": 2}}]}
        )
        self.assertEqual(phase_max_calls(phases, "pre"), {"post_message": 2})
        self.assertEqual(phase_max_calls(phases, "execution"), {})


class PerAgentToolBlockedTests(unittest.TestCase):
    def _env(self, blocked_tools, roles=None):
        env = SimpleNamespace(env_config={"blocked_tools": blocked_tools})
        if roles is not None:
            env.role_by_agent = roles
        return env

    def test_string_entry_applies_to_everyone(self):
        env = self._env({"planning": ["post_message"]})
        self.assertTrue(tool_blocked(env, "post_message", "planning", "A"))
        self.assertTrue(tool_blocked(env, "post_message", "planning", "B"))
        self.assertTrue(tool_blocked(env, "post_message", "planning", None))

    def test_agents_entry_scopes_by_name(self):
        env = self._env({"planning": [{"tool": "post_message", "agents": ["A"]}]})
        self.assertTrue(tool_blocked(env, "post_message", "planning", "A"))
        self.assertFalse(tool_blocked(env, "post_message", "planning", "B"))
        self.assertFalse(tool_blocked(env, "post_message", "planning", None))

    def test_agents_all(self):
        env = self._env({"planning": [{"tool": "post_message", "agents": "all"}]})
        self.assertTrue(tool_blocked(env, "post_message", "planning", "A"))
        self.assertTrue(tool_blocked(env, "post_message", "planning", None))

    def test_roles_entry_resolves_via_role_by_agent(self):
        env = self._env(
            {"planning": [{"tool": "post_message", "roles": ["adversary"]}]},
            roles={"A": "adversary", "B": "cooperative"},
        )
        self.assertTrue(tool_blocked(env, "post_message", "planning", "A"))
        self.assertFalse(tool_blocked(env, "post_message", "planning", "B"))

    def test_roles_entry_without_role_map_does_not_apply(self):
        env = self._env({"planning": [{"tool": "post_message", "roles": ["adversary"]}]})
        self.assertFalse(tool_blocked(env, "post_message", "planning", "A"))

    def test_wildcard_in_mapping_never_blocks_blackboard_tools(self):
        env = self._env({"pre_coordination": [{"tool": "*", "agents": "all"}]})
        self.assertFalse(tool_blocked(env, "post_message", "pre_coordination", "A"))
        self.assertTrue(tool_blocked(env, "schedule_patient", "pre_coordination", "A"))

    def test_mapping_without_tool_key_ignored(self):
        env = self._env({"planning": [{"agents": ["A"]}]})
        self.assertFalse(tool_blocked(env, "post_message", "planning", "A"))


class DiscoveryPerAgentTests(unittest.TestCase):
    def test_blackboard_schema_withheld_per_agent(self):
        env = SimpleNamespace(
            env_config={
                "blocked_tools": {
                    "planning": [{"tool": "post_message", "agents": ["A"]}]
                }
            }
        )
        discovery = ToolsetDiscovery()
        discovery.bind_environment(env)
        names_a = {
            t["function"]["name"]
            for t in discovery.get_tools_for_blackboard("planning", agent_name="A")
        }
        names_b = {
            t["function"]["name"]
            for t in discovery.get_tools_for_blackboard("planning", agent_name="B")
        }
        self.assertNotIn("post_message", names_a)
        self.assertIn("post_message", names_b)

    def test_unbound_discovery_unaffected(self):
        discovery = ToolsetDiscovery()
        names = {
            t["function"]["name"]
            for t in discovery.get_tools_for_blackboard("planning", agent_name="A")
        }
        self.assertIn("post_message", names)


class _StubEnv:
    """Minimal env for protocol-level tests."""

    def __init__(self, env_config, role_by_agent=None, secret_channel_by_agent=None):
        self.env_config = env_config
        if role_by_agent is not None:
            self.role_by_agent = role_by_agent
        if secret_channel_by_agent is not None:
            self.secret_channel_by_agent = secret_channel_by_agent


def _protocol(env, simulation_config=None):
    protocol = SequentialCommunicationProtocol.__new__(
        SequentialCommunicationProtocol
    )
    protocol.megaboard = Megaboard()
    protocol.environment = env
    protocol.environment_tools = None
    protocol._environment_tools_name = None
    protocol.simulation_config = simulation_config or {}
    protocol._current_turn_key = None
    protocol._tool_call_counts = {}
    return protocol


class ProtocolPerAgentGatingTests(unittest.IsolatedAsyncioTestCase):
    async def test_blackboard_tool_blocked_per_agent(self):
        env = _StubEnv(
            {"blocked_tools": {"planning": [{"tool": "post_message", "roles": ["adversary"]}]}},
            role_by_agent={"A": "adversary", "B": "cooperative"},
        )
        protocol = _protocol(env)
        protocol.megaboard.add_blackboard(["A", "B"])
        blocked = await protocol.blackboard_handle_tool_call(
            "post_message", "A", {"message": "hi"}, phase="planning", iteration=1
        )
        self.assertTrue(blocked.get("blocked"))
        ok = await protocol.blackboard_handle_tool_call(
            "post_message", "B", {"message": "hi"}, phase="planning", iteration=1
        )
        self.assertNotIn("blocked", ok)
        self.assertIn("event_id", ok)

    async def test_secret_channel_rewrite(self):
        env = _StubEnv(
            {
                "secret_channels": [
                    {"members": ["A", "B"], "enforce_in_phases": ["pre_coordination"]}
                ]
            },
            secret_channel_by_agent={"A": 1, "B": 1},
        )
        protocol = _protocol(env)
        protocol.megaboard.add_blackboard(["A", "B", "C"])
        secret_id = protocol.megaboard.add_blackboard(
            ["A", "B"], template={"private_channel": True, "visibility": "secret"}
        )
        env.secret_channel_by_agent = {"A": secret_id, "B": secret_id}

        self.assertEqual(secret_channel_rewrite(env, "A", "pre_coordination"), secret_id)
        self.assertIsNone(secret_channel_rewrite(env, "A", "planning"))
        self.assertIsNone(secret_channel_rewrite(env, "C", "pre_coordination"))

        result = await protocol.blackboard_handle_tool_call(
            "post_message",
            "A",
            {"message": "scheme", "blackboard_id": 0},
            phase="pre_coordination",
            iteration=0,
        )
        self.assertIn("event_id", result)
        secret_board = protocol.megaboard.blackboards[secret_id]
        self.assertTrue(any("scheme" in str(e) for e in secret_board.logs))
        public_board = protocol.megaboard.blackboards[0]
        self.assertFalse(any("scheme" in str(e) for e in public_board.logs))

    async def test_max_calls_cap(self):
        env = _StubEnv({})
        sim = {"phases": [{"name": "pre_coordination", "rounds": 2, "max_calls": {"post_message": 1}}]}
        protocol = _protocol(env, sim)
        protocol.megaboard.add_blackboard(["A"])
        protocol._current_turn_key = ("A", "pre_coordination", 1)

        first = await protocol.blackboard_handle_tool_call(
            "post_message", "A", {"message": "one"}, phase="pre_coordination", iteration=0
        )
        self.assertIn("event_id", first)
        second = await protocol.blackboard_handle_tool_call(
            "post_message", "A", {"message": "two"}, phase="pre_coordination", iteration=0
        )
        self.assertEqual(second.get("status"), "call_limit")
        self.assertTrue(second.get("blocked"))
        self.assertEqual(second, call_limit_response("post_message", "pre_coordination", 1, 1))
        protocol._current_turn_key = ("A", "pre_coordination", 2)
        third = await protocol.blackboard_handle_tool_call(
            "post_message", "A", {"message": "three"}, phase="pre_coordination", iteration=0
        )
        self.assertIn("event_id", third)


if __name__ == "__main__":
    unittest.main()
