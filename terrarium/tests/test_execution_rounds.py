"""Tests for execution-round mechanics: per-round schedule cap + seeded dice.

Pins the behavior added on feat/execution-rounds for the Hospital environment:

  - when a tool call carries a round number, a department may commit at most
    ``environment.max_schedules_per_round`` (default 1) schedule_patient
    bookings per round; failed calls do not consume the cap;
  - when two units claim the same (patient, step) within the same round, a
    dice roll from the run's seeded RNG decides the winner; the loser's booking
    is rolled back (slots, inventory, failure counters restored) and the
    challenger fails cleanly with no state mutation;
  - bookings with downstream steps already scheduled are never rolled back;
  - without a round number (legacy runners), the behavior is unchanged:
    first-come-first-served with a hard "already scheduled" rejection and no
    dice roll;
  - run_phase_entries passes the round number to agent_execution_turn and
    honors per-entry "shuffle_agents".

Runs without any LLM/API access, same pattern as
test_disallow_double_scheduling.py:

    PYTHONPATH=$PWD python -m unittest terrarium.tests.test_execution_rounds -v
"""

import copy
import unittest
from typing import Any, Dict, List, Optional
from unittest import mock

from terrarium.environments.dcops.hospital.hospital_env import HospitalEnvironment
from terrarium.environments.dcops.hospital.hospital_tools import HospitalTools
from terrarium.utils import run_phase_entries


class _ProtocolStub:
    environment = None


def _make_env(seed: int = 7, num_patients: int = 4, **env_overrides):
    env = HospitalEnvironment(
        communication_protocol=_ProtocolStub(),
        config={
            "simulation": {"seed": seed},
            "environment": {"num_hospitals": 2, "num_patients": num_patients, **env_overrides},
        },
        tool_logger=None,
    )
    tools = HospitalTools(blackboard_manager=None, environment=env)
    return env, tools


def _schedule(tools, agent, patient_id, step_index, start_time, round_num=None) -> Dict[str, Any]:
    return tools.handle_tool_call(
        "schedule_patient",
        agent,
        {
            "patient_id": patient_id,
            "step_index": step_index,
            "start_time": start_time,
            "rationale": "test",
        },
        phase="execution",
        iteration=1,
        round_num=round_num,
    )


def _triage_patients(env) -> List[str]:
    return [
        p_id
        for p_id, patient in env.patients.items()
        if patient["pathway"][0]["service"] == "Triage"
    ]


TRIAGE_A = "General_Hospital_Triage"
TRIAGE_B = "St_Marys_Center_Triage"


class TestPerRoundScheduleCap(unittest.TestCase):
    def test_second_booking_same_round_is_rejected(self):
        env, tools = _make_env()
        p1, p2 = _triage_patients(env)[:2]
        start1 = int(env.patients[p1]["arrival_time"])
        start2 = int(env.patients[p2]["arrival_time"])

        first = _schedule(tools, TRIAGE_A, p1, 0, start1, round_num=1)
        self.assertEqual(first["status"], "success")

        second = _schedule(tools, TRIAGE_A, p2, 0, start2, round_num=1)
        self.assertEqual(second["status"], "failed")
        self.assertIn("limit 1 per round", second["reason"])
        self.assertNotIn(0, env.patient_states[p2]["scheduled_steps"])

    def test_cap_resets_next_round(self):
        env, tools = _make_env()
        p1, p2 = _triage_patients(env)[:2]
        _schedule(tools, TRIAGE_A, p1, 0, int(env.patients[p1]["arrival_time"]), round_num=1)
        second = _schedule(
            tools, TRIAGE_A, p2, 0, int(env.patients[p2]["arrival_time"]), round_num=2
        )
        self.assertEqual(second["status"], "success")
        self.assertEqual(env.patient_states[p2]["scheduled_steps"][0]["agent"], TRIAGE_A)
        self.assertEqual(env.patient_states[p2]["scheduled_steps"][0]["round"], 2)

    def test_failed_call_does_not_consume_cap(self):
        env, tools = _make_env()
        p1 = _triage_patients(env)[0]
        start = int(env.patients[p1]["arrival_time"])

        bad = _schedule(tools, TRIAGE_A, "Patient_999", 0, start, round_num=1)
        self.assertEqual(bad["status"], "failed")
        good = _schedule(tools, TRIAGE_A, p1, 0, start, round_num=1)
        self.assertEqual(good["status"], "success")

    def test_cap_is_configurable(self):
        env, tools = _make_env(max_schedules_per_round=2)
        p1, p2 = _triage_patients(env)[:2]
        r1 = _schedule(tools, TRIAGE_A, p1, 0, int(env.patients[p1]["arrival_time"]), round_num=1)
        r2 = _schedule(tools, TRIAGE_A, p2, 0, int(env.patients[p2]["arrival_time"]), round_num=1)
        self.assertEqual(r1["status"], "success")
        self.assertEqual(r2["status"], "success")

    def test_no_round_number_means_no_cap(self):
        env, tools = _make_env()
        p1, p2 = _triage_patients(env)[:2]
        r1 = _schedule(tools, TRIAGE_A, p1, 0, int(env.patients[p1]["arrival_time"]))
        r2 = _schedule(tools, TRIAGE_A, p2, 0, int(env.patients[p2]["arrival_time"]))
        self.assertEqual(r1["status"], "success")
        self.assertEqual(r2["status"], "success")


class TestSameRoundConflictDice(unittest.TestCase):
    def test_exactly_one_winner_and_rollback_when_challenger_wins(self):
        env, tools = _make_env()
        p_id = _triage_patients(env)[0]
        start = int(env.patients[p_id]["arrival_time"])

        inventory_a_before = copy.deepcopy(env.inventory["General_Hospital"])
        failures_before = dict(env.resource_failures)
        first = _schedule(tools, TRIAGE_A, p_id, 0, start, round_num=1)
        self.assertEqual(first["status"], "success")

        # Force the dice to pick the challenger (second element of [incumbent, challenger]).
        with mock.patch("random.choice", side_effect=lambda seq: seq[-1]):
            second = _schedule(tools, TRIAGE_B, p_id, 0, start, round_num=1)

        self.assertEqual(second["status"], "success")
        entry = env.patient_states[p_id]["scheduled_steps"][0]
        self.assertEqual(entry["agent"], TRIAGE_B)
        # Incumbent's slot was released.
        self.assertNotIn(p_id, env.schedule[TRIAGE_A].get(start, []))
        self.assertIn(p_id, env.schedule[TRIAGE_B].get(start, []))
        # Incumbent's hospital inventory and failure counters were restored.
        self.assertEqual(env.inventory["General_Hospital"], inventory_a_before)
        self.assertEqual(env.resource_failures, failures_before)

    def test_loser_fails_without_state_mutation(self):
        env, tools = _make_env()
        p_id = _triage_patients(env)[0]
        start = int(env.patients[p_id]["arrival_time"])

        _schedule(tools, TRIAGE_A, p_id, 0, start, round_num=1)
        inventory_b_before = copy.deepcopy(env.inventory["St_Marys_Center"])
        schedule_b_before = copy.deepcopy(env.schedule[TRIAGE_B])
        reward_before = env.get_final_summary()["joint_reward"]

        # Force the dice to keep the incumbent.
        with mock.patch("random.choice", side_effect=lambda seq: seq[0]):
            second = _schedule(tools, TRIAGE_B, p_id, 0, start, round_num=1)

        self.assertEqual(second["status"], "failed")
        self.assertIn("dice", second["reason"].lower())
        self.assertEqual(env.patient_states[p_id]["scheduled_steps"][0]["agent"], TRIAGE_A)
        self.assertEqual(env.schedule[TRIAGE_B], schedule_b_before)
        self.assertEqual(env.inventory["St_Marys_Center"], inventory_b_before)
        self.assertEqual(env.get_final_summary()["joint_reward"], reward_before)

    def test_dice_is_reproducible_for_a_given_seed(self):
        winners = []
        for _ in range(2):
            env, tools = _make_env(seed=123)
            p_id = _triage_patients(env)[0]
            start = int(env.patients[p_id]["arrival_time"])
            _schedule(tools, TRIAGE_A, p_id, 0, start, round_num=1)
            _schedule(tools, TRIAGE_B, p_id, 0, start, round_num=1)
            winners.append(env.patient_states[p_id]["scheduled_steps"][0]["agent"])
        self.assertEqual(winners[0], winners[1])
        self.assertIn(winners[0], {TRIAGE_A, TRIAGE_B})

    def test_downstream_bookings_protect_incumbent(self):
        env, tools = _make_env()
        p_id = _triage_patients(env)[0]
        patient = env.patients[p_id]
        start0 = int(patient["arrival_time"])

        _schedule(tools, TRIAGE_A, p_id, 0, start0, round_num=1)
        # Same-hospital radiology books step 1 in the same round.
        if len(patient["pathway"]) > 1 and patient["pathway"][1]["service"] == "Radiology":
            step1 = patient["pathway"][1]
            r = _schedule(
                tools, "General_Hospital_Radiology", p_id, 1,
                start0 + 1, round_num=1,
            )
            self.assertEqual(r["status"], "success")
            dur0 = patient["pathway"][0]["duration"]
            self.assertEqual(start0 + 1, start0 + dur0)  # no transfer penalty, same hospital
            self.assertEqual(step1["duration"], 2)

            # Challenger cannot dislodge the incumbent now that step 1 exists.
            with mock.patch("random.choice") as choice_mock:
                second = _schedule(tools, TRIAGE_B, p_id, 0, start0, round_num=1)
            choice_mock.assert_not_called()
            self.assertEqual(second["status"], "failed")
            self.assertIn("already scheduled", second["reason"])
            self.assertEqual(env.patient_states[p_id]["scheduled_steps"][0]["agent"], TRIAGE_A)
            self.assertIn(1, env.patient_states[p_id]["scheduled_steps"])

    def test_legacy_no_round_hard_rejects_without_dice(self):
        env, tools = _make_env()
        p_id = _triage_patients(env)[0]
        start = int(env.patients[p_id]["arrival_time"])

        _schedule(tools, TRIAGE_A, p_id, 0, start)
        with mock.patch("random.choice") as choice_mock:
            second = _schedule(tools, TRIAGE_B, p_id, 0, start)
        choice_mock.assert_not_called()
        self.assertEqual(second["status"], "failed")
        self.assertIn("already scheduled", second["reason"])
        self.assertEqual(env.patient_states[p_id]["scheduled_steps"][0]["agent"], TRIAGE_A)

    def test_dice_can_be_disabled_by_config(self):
        env, tools = _make_env(schedule_conflict_dice=False)
        p_id = _triage_patients(env)[0]
        start = int(env.patients[p_id]["arrival_time"])

        _schedule(tools, TRIAGE_A, p_id, 0, start, round_num=1)
        with mock.patch("random.choice") as choice_mock:
            second = _schedule(tools, TRIAGE_B, p_id, 0, start, round_num=1)
        choice_mock.assert_not_called()
        self.assertEqual(second["status"], "failed")
        self.assertIn("already scheduled", second["reason"])


class _Agent:
    def __init__(self, name: str) -> None:
        self.name = name


class _Environment:
    def __init__(self, agents: List[_Agent]) -> None:
        self.agents = agents

    def build_agent_context(self, agent_name, phase, iteration, **kwargs) -> Dict[str, Any]:
        return {"agent_name": agent_name, "phase": phase, "iteration": iteration, **kwargs}


class _ExecutionProtocol:
    """Records agent_execution_turn calls (no generic agent_turn)."""

    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    async def agent_execution_turn(
        self, agent, agent_name, agent_context, environment, iteration, round_num=None
    ):
        self.calls.append(
            {"agent_name": agent_name, "iteration": iteration, "round_num": round_num}
        )
        return {"phase": "execution", "agent_name": agent_name}


class TestRunPhaseEntriesRounds(unittest.IsolatedAsyncioTestCase):
    async def test_execution_rounds_pass_round_num_and_shuffle_per_round(self):
        agents = [_Agent(f"agent_{i}") for i in range(4)]
        env = _Environment(agents)
        protocol = _ExecutionProtocol()

        turns = await run_phase_entries(
            [{"phase": "execution", "rounds": 3, "shuffle_agents": True}],
            environment=env,
            communication_protocol=protocol,
            iteration=1,
            progress=False,
        )

        self.assertEqual(len(turns), 12)
        self.assertEqual([t["round"] for t in turns], [1] * 4 + [2] * 4 + [3] * 4)
        self.assertEqual(
            [c["round_num"] for c in protocol.calls], [1] * 4 + [2] * 4 + [3] * 4
        )
        # Each round visits every agent exactly once (shuffle = permutation).
        for r in (1, 2, 3):
            names = [t["agent_name"] for t in turns if t["round"] == r]
            self.assertEqual(sorted(names), sorted(a.name for a in agents))

    async def test_without_shuffle_order_is_stable(self):
        agents = [_Agent(f"agent_{i}") for i in range(3)]
        env = _Environment(agents)
        protocol = _ExecutionProtocol()

        turns = await run_phase_entries(
            [{"phase": "execution", "rounds": 2}],
            environment=env,
            communication_protocol=protocol,
            iteration=1,
            progress=False,
        )

        self.assertEqual(
            [t["agent_name"] for t in turns],
            ["agent_0", "agent_1", "agent_2"] * 2,
        )


if __name__ == "__main__":
    unittest.main()
