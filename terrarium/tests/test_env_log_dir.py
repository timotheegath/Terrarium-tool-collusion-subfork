"""Regression tests: environment iteration logs land in the run log dir.

`AbstractEnvironment._resolve_log_dir` prefers the attached tool logger's
``log_dir`` — experiment runners point it at the run dir under
``outputs/.../runs/<model>/<sweep>/<run_id>/`` — so `data_iteration_*.json`
and `patients.json` must not leak into a repo-root ``logs/`` tree. Without a
tool logger it falls back to
``<logging.root_dir>/<env>/<tag_model>/<timestamp>/seed_<seed>`` (default
root ``logs/``).

Runs without any LLM/API access: the environment is instantiated directly with
a stub protocol (same pattern as test_disallow_double_scheduling.py).

Run from the repo root with the local package shadowing any installed copy:

    PYTHONPATH=$PWD <python> -m unittest terrarium.tests.test_env_log_dir -v
"""

import json
import random
import tempfile
import unittest
from pathlib import Path

from terrarium.core.logger import ToolCallLogger
from terrarium.environments.dcops.hospital.hospital_env import HospitalEnvironment


class _ProtocolStub:
    environment = None


def _base_cfg(**extra):
    cfg = {
        "simulation": {"seed": 3, "run_timestamp": "20260101-000000__run", "tags": ["mytag"]},
        "environment": {"num_hospitals": 2, "num_patients": 3},
    }
    cfg.update(extra)
    return cfg


def _make_env(cfg, tool_logger):
    state = random.getstate()
    try:
        return HospitalEnvironment(
            communication_protocol=_ProtocolStub(),
            config=cfg,
            tool_logger=tool_logger,
        )
    finally:
        random.setstate(state)


class EnvLogDirTests(unittest.TestCase):
    def test_log_iteration_writes_into_tool_logger_log_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run_dir"
            cfg = _base_cfg()
            tool_logger = ToolCallLogger(
                environment_name="HospitalEnvironment",
                seed=3,
                config=cfg,
                log_dir=run_dir,
            )
            env = _make_env(cfg, tool_logger)
            env.log_iteration(1)

            self.assertTrue((run_dir / "data_iteration_1.json").is_file())
            self.assertTrue((run_dir / "patients.json").is_file())
            entry = json.loads((run_dir / "data_iteration_1.json").read_text())
            self.assertEqual(entry["iteration"], 1)

    def test_tool_logger_log_dir_takes_precedence_over_logging_root_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run_dir"
            fallback_root = Path(tmp) / "fallback"
            cfg = _base_cfg(logging={"root_dir": str(fallback_root)})
            tool_logger = ToolCallLogger(
                environment_name="HospitalEnvironment",
                seed=3,
                config=cfg,
                log_dir=run_dir,
            )
            env = _make_env(cfg, tool_logger)
            env.log_iteration(1)

            self.assertTrue((run_dir / "data_iteration_1.json").is_file())
            self.assertFalse(fallback_root.exists())

    def test_fallback_uses_logging_root_dir_and_tag_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "custom_logs"
            cfg = _base_cfg(logging={"root_dir": str(root)})
            env = _make_env(cfg, tool_logger=None)
            env.log_iteration(1)

            # tag_model = "mytag_unknown" (no llm block -> model "unknown")
            expected = (
                root
                / "HospitalEnvironment"
                / "mytag_unknown"
                / "20260101-000000__run"
                / "seed_3"
            )
            self.assertTrue((expected / "data_iteration_1.json").is_file())
            self.assertTrue((expected / "patients.json").is_file())


if __name__ == "__main__":
    unittest.main()
