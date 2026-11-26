import unittest
from typing import Any, Dict, List, Optional

from terrarium.agents.base import BaseAgent
from terrarium.communication_protocols.local import (
    LocalCommunicationProtocol,
    PhaseToolPolicy,
)
from terrarium.environments.dcops.hospital.hospital_env import HospitalEnvironment
from terrarium.tools.discovery import ToolsetDiscovery


class _ProtocolStub:
    environment = None


class _DummyJiraEnvironment:
    tools_environment_name = "JiraTicketEnvironment"

    def __init__(self):
        self.tasks = {"ISSUE-1": {"summary": "Fix the failing test"}}
        self.assignment = {}
        self.agent_names = ["Alex"]

    def joint_reward(self, assignment):
        return float(sum(1 for task_id in assignment.values() if task_id is not None))


class _RecordingAgent:
    def __init__(self):
        self.calls: List[Dict[str, Any]] = []

    async def generate_response(self, **kwargs):
        self.calls.append(kwargs)
        return {"phase": kwargs["phase"], "round_num": kwargs["round_num"]}


class _EnvStub:
    prompts = object()


def _make_protocol() -> LocalCommunicationProtocol:
    return LocalCommunicationProtocol(config={})


def _tool_names(tools: List[Dict[str, Any]]) -> List[str]:
    return [t["function"]["name"] for t in tools]


class PlanningShapedPhaseSetTests(unittest.TestCase):
    def test_default_phase_set_is_planning_only(self):
        discovery = ToolsetDiscovery()
        self.assertEqual(discovery.planning_shaped_phases, {"planning"})
        self.assertEqual(ToolsetDiscovery.planning_shaped_phases, {"planning"})
        self.assertEqual(_tool_names(discovery.get_tools_for_blackboard("planning")), ["post_message"])
        self.assertEqual(discovery.get_tools_for_blackboard("execution"), [])
        self.assertEqual(discovery.get_tools_for_blackboard("pre_coordination"), [])

    def test_instance_override_adds_planning_shaped_phase(self):
        discovery = ToolsetDiscovery()
        discovery.planning_shaped_phases = {"planning", "pre_coordination"}
        self.assertEqual(
            _tool_names(discovery.get_tools_for_blackboard("pre_coordination")),
            ["post_message"],
        )
        # The class attribute is untouched by the instance override.
        self.assertEqual(ToolsetDiscovery.planning_shaped_phases, {"planning"})
        self.assertEqual(ToolsetDiscovery().get_tools_for_blackboard("pre_coordination"), [])


class AllowedToolsTests(unittest.TestCase):
    def test_allowed_tools_filters_blackboard_tool_names(self):
        discovery = ToolsetDiscovery()
        self.assertEqual(
            discovery.get_blackboard_tool_names(retrieval_enabled=True),
            {"post_message", "recall"},
        )
        self.assertEqual(
            discovery.get_blackboard_tool_names(
                retrieval_enabled=True, allowed_tools={"post_message"}
            ),
            {"post_message"},
        )
        self.assertEqual(
            discovery.get_blackboard_tool_names(allowed_tools=set()), set()
        )

    def test_allowed_tools_filters_blackboard_tools(self):
        discovery = ToolsetDiscovery()
        self.assertEqual(
            _tool_names(
                discovery.get_tools_for_blackboard(
                    "planning", retrieval_enabled=True, allowed_tools={"recall"}
                )
            ),
            ["recall"],
        )
        self.assertEqual(
            discovery.get_tools_for_blackboard("planning", allowed_tools=set()), []
        )
        self.assertEqual(
            _tool_names(
                discovery.get_tools_for_blackboard("planning", allowed_tools={"post_message"})
            ),
            ["post_message"],
        )

    def test_allowed_tools_does_not_affect_other_phases(self):
        discovery = ToolsetDiscovery()
        self.assertEqual(
            discovery.get_tools_for_blackboard("execution", allowed_tools={"post_message"}),
            [],
        )


class PhaseToolPolicyDataTests(unittest.TestCase):
    def test_defaults(self):
        policy = PhaseToolPolicy()
        self.assertTrue(policy.environment_tools_offered)
        self.assertFalse(policy.block_environment_tool_calls)
        self.assertIsNone(policy.blackboard_tools)
        self.assertIsNone(policy.route_post_message_to)
        self.assertIsNone(policy.max_post_messages_per_agent)

    def test_push_pop_and_isolation_between_instances(self):
        first = _make_protocol()
        second = _make_protocol()
        self.assertIsNone(first.active_phase_policy("pre_coordination"))

        policy = PhaseToolPolicy(blackboard_tools={"post_message"})
        first.set_phase_policy("pre_coordination", policy)
        self.assertIs(first.active_phase_policy("pre_coordination"), policy)
        # Policies are per instance: the second protocol is unaffected.
        self.assertIsNone(second.active_phase_policy("pre_coordination"))

        popped = first.pop_phase_policy("pre_coordination")
        self.assertIs(popped, policy)
        self.assertIsNone(first.active_phase_policy("pre_coordination"))
        self.assertIsNone(first.pop_phase_policy("pre_coordination"))

    def test_active_phase_policy_none_phase(self):
        protocol = _make_protocol()
        protocol.set_phase_policy("pre_coordination", PhaseToolPolicy())
        self.assertIsNone(protocol.active_phase_policy(None))


class PhaseToolPolicyEnforcementTests(unittest.IsolatedAsyncioTestCase):
    async def test_blocked_environment_tool_calls_do_not_execute(self):
        protocol = _make_protocol()
        protocol.set_phase_policy(
            "pre_coordination", PhaseToolPolicy(block_environment_tool_calls=True)
        )
        env = _DummyJiraEnvironment()
        protocol.environment = env

        result = await protocol.environment_handle_tool_call(
            "assign_task",
            "Alex",
            {"task_id": "ISSUE-1"},
            phase="pre_coordination",
            iteration=0,
        )
        self.assertTrue(result["blocked"])
        self.assertIsNone(result["result"])
        self.assertEqual(env.assignment, {})
        self.assertEqual(protocol.tool_events[-1].phase, "pre_coordination")
        self.assertTrue(protocol.tool_events[-1].result["blocked"])

    async def test_environment_tool_calls_execute_without_policy(self):
        protocol = _make_protocol()
        protocol.megaboard.log_action_to_blackboards = lambda *args, **kwargs: None
        env = _DummyJiraEnvironment()
        protocol.environment = env

        result = await protocol.environment_handle_tool_call(
            "assign_task",
            "Alex",
            {"task_id": "ISSUE-1"},
            phase="execution",
            iteration=1,
        )
        self.assertEqual(result["status"], "success")
        self.assertEqual(env.assignment, {"Alex": "ISSUE-1"})

    async def test_post_message_routed_to_secret_board(self):
        protocol = _make_protocol()
        public_id = protocol.megaboard.add_blackboard(["Alex", "Bob"])
        secret_id = protocol.megaboard.add_blackboard(
            ["Alex", "Bob"], template={"secret_channel": True}
        )
        protocol.set_phase_policy(
            "pre_coordination", PhaseToolPolicy(route_post_message_to=secret_id)
        )

        result = await protocol.blackboard_handle_tool_call(
            "post_message",
            "Alex",
            {"message": "secret hello", "blackboard_id": public_id},
            phase="pre_coordination",
            iteration=0,
        )
        self.assertIn("event_id", result)
        self.assertEqual(protocol.tool_events[-1].arguments["blackboard_id"], secret_id)
        self.assertEqual(protocol.megaboard.blackboards[public_id].logs, [])
        self.assertEqual(
            protocol.megaboard.blackboards[secret_id].logs[-1]["payload"]["content"],
            "secret hello",
        )

    async def test_post_message_routing_when_blackboard_id_missing_or_invalid(self):
        protocol = _make_protocol()
        protocol.megaboard.add_blackboard(["Alex", "Bob"])
        secret_id = protocol.megaboard.add_blackboard(
            ["Alex", "Bob"], template={"secret_channel": True}
        )
        protocol.set_phase_policy(
            "pre_coordination", PhaseToolPolicy(route_post_message_to=secret_id)
        )

        await protocol.blackboard_handle_tool_call(
            "post_message",
            "Alex",
            {"message": "no board"},
            phase="pre_coordination",
            iteration=0,
        )
        self.assertEqual(protocol.tool_events[-1].arguments["blackboard_id"], secret_id)

        await protocol.blackboard_handle_tool_call(
            "post_message",
            "Alex",
            {"message": "bad board", "blackboard_id": "not-a-number"},
            phase="pre_coordination",
            iteration=0,
        )
        self.assertEqual(protocol.tool_events[-1].arguments["blackboard_id"], secret_id)

    async def test_post_message_not_routed_without_policy(self):
        protocol = _make_protocol()
        public_id = protocol.megaboard.add_blackboard(["Alex", "Bob"])

        await protocol.blackboard_handle_tool_call(
            "post_message",
            "Alex",
            {"message": "public hello", "blackboard_id": public_id},
            phase="planning",
            iteration=1,
        )
        self.assertEqual(protocol.tool_events[-1].arguments["blackboard_id"], public_id)
        self.assertEqual(
            protocol.megaboard.blackboards[public_id].logs[-1]["payload"]["content"],
            "public hello",
        )

    async def test_per_agent_cap_and_round_reset(self):
        protocol = _make_protocol()
        protocol.megaboard.add_blackboard(["Alex", "Bob"])
        protocol.set_phase_policy(
            "pre_coordination",
            PhaseToolPolicy(max_post_messages_per_agent=1),
        )

        first = await protocol.blackboard_handle_tool_call(
            "post_message",
            "Alex",
            {"message": "one", "blackboard_id": 0},
            phase="pre_coordination",
            iteration=0,
        )
        self.assertIn("event_id", first)
        self.assertEqual(protocol.get_post_message_count("Alex"), 1)

        second = await protocol.blackboard_handle_tool_call(
            "post_message",
            "Alex",
            {"message": "two", "blackboard_id": 0},
            phase="pre_coordination",
            iteration=0,
        )
        self.assertTrue(second["blocked"])
        self.assertEqual(second["limit"], 1)
        self.assertEqual(second["count"], 1)

        # The cap is per agent: Bob still has its own budget.
        bob = await protocol.blackboard_handle_tool_call(
            "post_message",
            "Bob",
            {"message": "bob one", "blackboard_id": 0},
            phase="pre_coordination",
            iteration=0,
        )
        self.assertIn("event_id", bob)

        # Round boundary: resetting the counters gives every agent a fresh budget.
        protocol.reset_post_message_counts()
        self.assertEqual(protocol.get_post_message_count("Alex"), 0)
        third = await protocol.blackboard_handle_tool_call(
            "post_message",
            "Alex",
            {"message": "three", "blackboard_id": 0},
            phase="pre_coordination",
            iteration=0,
        )
        self.assertIn("event_id", third)

    async def test_zero_cap_rejects_all_post_messages(self):
        protocol = _make_protocol()
        protocol.megaboard.add_blackboard(["Alex"])
        protocol.set_phase_policy(
            "pre_coordination", PhaseToolPolicy(max_post_messages_per_agent=0)
        )

        result = await protocol.blackboard_handle_tool_call(
            "post_message",
            "Alex",
            {"message": "blocked", "blackboard_id": 0},
            phase="pre_coordination",
            iteration=0,
        )
        self.assertTrue(result["blocked"])
        self.assertEqual(result["limit"], 0)
        self.assertEqual(result["count"], 0)
        self.assertEqual(protocol.megaboard.blackboards[0].logs, [])

    async def test_blackboard_tools_allowlist_rejects_withheld_tools(self):
        protocol = _make_protocol()
        protocol.megaboard.add_blackboard(["Alex"])
        protocol.set_phase_policy(
            "pre_coordination", PhaseToolPolicy(blackboard_tools=set())
        )

        result = await protocol.blackboard_handle_tool_call(
            "post_message",
            "Alex",
            {"message": "blocked", "blackboard_id": 0},
            phase="pre_coordination",
            iteration=0,
        )
        self.assertTrue(result["blocked"])
        self.assertEqual(protocol.megaboard.blackboards[0].logs, [])

    async def test_verbatim_phase_label_in_turn_dispatch(self):
        protocol = _make_protocol()
        protocol.megaboard.add_blackboard(["Alex"])
        agent = _RecordingAgent()

        result = await protocol.agent_turn(
            agent,
            "Alex",
            {"agent_name": "Alex"},
            _EnvStub(),
            0,
            phase="pre_coordination",
            round_num=2,
        )
        self.assertEqual(result["phase"], "pre_coordination")
        self.assertEqual(agent.calls[0]["phase"], "pre_coordination")
        self.assertEqual(agent.calls[0]["round_num"], 2)

        await protocol.blackboard_handle_tool_call(
            "post_message",
            "Alex",
            {"message": "hi", "blackboard_id": 0},
            phase="pre_coordination",
            iteration=0,
        )
        self.assertEqual(protocol.tool_events[-1].phase, "pre_coordination")


class AgentPhasePolicyWiringTests(unittest.TestCase):
    def test_set_meta_context_records_active_policy(self):
        protocol = _make_protocol()
        policy = PhaseToolPolicy(blackboard_tools={"post_message"})
        protocol.set_phase_policy("pre_coordination", policy)

        agent = BaseAgent(client=object(), name="Alex", environment_name="HospitalEnvironment")
        agent.communication_protocol = protocol
        agent.set_meta_context("Alex", "pre_coordination", 0, 1)

        self.assertIs(agent.active_phase_policy, policy)
        self.assertEqual(agent._allowed_blackboard_tools(), {"post_message"})

        agent.toolset_discovery.planning_shaped_phases = {"planning", "pre_coordination"}
        self.assertEqual(
            _tool_names(
                agent.toolset_discovery.get_tools_for_blackboard(
                    "pre_coordination", allowed_tools=agent._allowed_blackboard_tools()
                )
            ),
            ["post_message"],
        )
        self.assertEqual(
            agent.toolset_discovery.get_tools_for_blackboard(
                "pre_coordination", allowed_tools=set()
            ),
            [],
        )

    def test_no_policy_means_no_allowlist(self):
        protocol = _make_protocol()
        agent = BaseAgent(client=object(), name="Alex", environment_name="HospitalEnvironment")
        agent.communication_protocol = protocol
        agent.set_meta_context("Alex", "planning", 1, 1)

        self.assertIsNone(agent.active_phase_policy)
        self.assertIsNone(agent._allowed_blackboard_tools())
        self.assertEqual(
            _tool_names(agent.toolset_discovery.get_tools_for_blackboard("planning")),
            ["post_message"],
        )


class HospitalPreCoordinationContextTests(unittest.TestCase):
    def _make_env(self) -> HospitalEnvironment:
        return HospitalEnvironment(
            communication_protocol=_ProtocolStub(),
            config={
                "simulation": {"seed": 7},
                "environment": {"num_hospitals": 2, "num_patients": 3},
            },
            tool_logger=None,
        )

    def test_pre_coordination_context_is_minimal(self):
        env = self._make_env()
        agent_name = next(a for a in env.agent_names if a != "Resource_Provisioner")

        context = env.build_agent_context(
            agent_name,
            phase="pre_coordination",
            iteration=0,
            round_number=1,
            misalignment_role="adversary",
            misalignment_adversaries=[agent_name],
            misalignment_secret_blackboard_id=1,
            post_message_available=True,
            inventory_snapshot="must not leak",
        )

        self.assertEqual(context["agent_name"], agent_name)
        self.assertEqual(context["phase"], "pre_coordination")
        self.assertEqual(context["iteration"], 0)
        self.assertEqual(context["role"], "department")
        self.assertEqual(context["misalignment_role"], "adversary")
        self.assertEqual(context["misalignment_secret_blackboard_id"], 1)
        self.assertTrue(context["post_message_available"])
        for leaked in (
            "inventory_snapshot",
            "failures_so_far",
            "failures_by_hospital",
            "job_queue",
            "cost_table",
            "dept_info",
            "round_number",
        ):
            self.assertNotIn(leaked, context)

    def test_pre_coordination_context_provisioner_role(self):
        env = self._make_env()
        context = env.build_agent_context("Resource_Provisioner", "pre_coordination", 0)
        self.assertEqual(context["role"], "provisioner")
        self.assertNotIn("inventory_snapshot", context)

    def test_other_phases_keep_full_context(self):
        env = self._make_env()
        agent_name = next(a for a in env.agent_names if a != "Resource_Provisioner")
        context = env.build_agent_context(agent_name, "planning", 1)
        self.assertIn("job_queue", context)
        self.assertIn("cost_table", context)
        self.assertIn("dept_info", context)


if __name__ == "__main__":
    unittest.main()
