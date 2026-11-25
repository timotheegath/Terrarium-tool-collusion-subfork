"""Regression tests for the broadcast_message rationale label.

Pins the fix on feat/hospital-broadcast-rationale-label: the broadcast
rationale is part of the public, auditable blackboard record, so
HospitalTools.execute_action logs it with the same plain "Rationale:" label
used by schedule_patient and transfer_resources -- never as
"(Internal Rationale: ...)" -- and the broadcast_message tool schema describes
it as a recorded, public note rather than an internal one.

Runs without any LLM/API access: the environment is instantiated directly with
a stub protocol (same pattern as test_disallow_double_scheduling.py) and driven
through the real tool entry point (HospitalTools.handle_tool_call), with a
recording blackboard-manager stub capturing exactly what would be posted to the
blackboards via log_action_to_blackboards.

Run from the repo root with the local package shadowing any installed copy:

    PYTHONPATH=$PWD <python> -m unittest terrarium.tests.test_broadcast_rationale_label -v
    # or, if pytest is available:
    PYTHONPATH=$PWD <python> -m pytest terrarium/tests/test_broadcast_rationale_label.py -v
"""

import unittest
from typing import Any, Dict, List

from terrarium.environments.dcops.hospital.hospital_env import HospitalEnvironment
from terrarium.environments.dcops.hospital.hospital_tools import HospitalTools


class _ProtocolStub:
    environment = None


class _RecordingBlackboardManager:
    """Stands in for BlackboardManager; records every logged action."""

    def __init__(self):
        self.logged: List[Dict[str, Any]] = []

    def log_action_to_blackboards(self, agent_name, action, result, phase=None, iteration=None):
        self.logged.append(
            {
                "agent_name": agent_name,
                "action": action,
                "result": result,
                "phase": phase,
                "iteration": iteration,
            }
        )


def _make_tools():
    env = HospitalEnvironment(
        communication_protocol=_ProtocolStub(),
        config={
            "simulation": {"seed": 7},
            "environment": {"num_hospitals": 2, "num_patients": 3},
        },
        tool_logger=None,
    )
    blackboard_manager = _RecordingBlackboardManager()
    tools = HospitalTools(blackboard_manager=blackboard_manager, environment=env)
    return tools, blackboard_manager


class TestBroadcastRationaleLabel(unittest.TestCase):
    def test_broadcast_result_uses_public_rationale_label(self):
        tools, _ = _make_tools()
        result = tools.handle_tool_call(
            "broadcast_message",
            "General_Hospital_Triage",
            {"message": "Requesting extra PPE", "rationale": "We are running low on PPE."},
            phase="execution",
            iteration=1,
        )

        self.assertEqual(result["status"], "success")
        self.assertIn("Rationale: We are running low on PPE.", result["result"])
        self.assertNotIn("(Internal Rationale:", result["result"])
        self.assertNotIn("Internal", result["result"])

    def test_blackboard_receives_same_public_rationale_string(self):
        tools, blackboard_manager = _make_tools()
        result = tools.handle_tool_call(
            "broadcast_message",
            "General_Hospital_Triage",
            {"message": "Status update", "rationale": "Two patients triaged."},
            phase="execution",
            iteration=1,
        )

        self.assertEqual(len(blackboard_manager.logged), 1)
        logged_result = blackboard_manager.logged[0]["result"]
        self.assertEqual(logged_result["result"], result["result"])
        self.assertIn("Rationale: Two patients triaged.", logged_result["result"])
        self.assertNotIn("(Internal Rationale:", logged_result["result"])

    def test_broadcast_schema_describes_rationale_as_public(self):
        tools, _ = _make_tools()
        broadcast = next(
            t for t in tools.get_tools("execution")
            if t["function"]["name"] == "broadcast_message"
        )
        description = broadcast["function"]["parameters"]["properties"]["rationale"]["description"]
        self.assertNotIn("Internal", description)


if __name__ == "__main__":
    unittest.main()
