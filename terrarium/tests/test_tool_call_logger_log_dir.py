"""Branch-local tests for feat/tool-call-logger-log-dir.

Pins the new optional ``log_dir`` parameter of ``ToolCallLogger``:

- when ``log_dir`` is passed (as a ``Path`` or ``str``), tool call artifacts
  are written directly into that directory, so runners can point the logger at
  an arbitrary run directory instead of the forced
  logs/<env>/<tag_model>/<timestamp>/seed_<seed> layout;
- when ``log_dir`` is omitted, the standard layout is still produced.

Runs without any LLM/API access.

Run from the repo root with the local package shadowing any installed copy:

    PYTHONPATH=$PWD <python> -m unittest terrarium.tests.test_tool_call_logger_log_dir -v
    # or, if pytest is available:
    PYTHONPATH=$PWD <python> -m pytest terrarium/tests/test_tool_call_logger_log_dir.py -v
"""

import json
import os
import tempfile
import unittest
from pathlib import Path

from terrarium.core.logger import ToolCallLogger
from terrarium.utils import build_log_dir, get_tag_model_subdir


class ToolCallLoggerLogDirTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        self._cwd = os.getcwd()
        os.chdir(self.tmp)
        self.addCleanup(os.chdir, self._cwd)

    def test_explicit_log_dir_path_is_used(self):
        run_dir = self.tmp / "run_42"
        logger = ToolCallLogger("Hospital", seed=3, log_dir=run_dir)

        self.assertEqual(logger.log_dir, run_dir)
        self.assertTrue((run_dir / "tool_calls.json").is_file())

        logger.log_tool_call(
            "agent_a",
            "execution",
            "schedule_patient",
            {"patient_id": "p1"},
            {"success": True},
            iteration=1,
            duration_ms=12.5,
        )
        entries = json.loads((run_dir / "tool_calls.json").read_text())
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["tool_name"], "schedule_patient")
        self.assertEqual(entries[0]["agent_name"], "agent_a")
        self.assertTrue(entries[0]["success"])

    def test_explicit_log_dir_accepts_str(self):
        run_dir = self.tmp / "run_str"
        logger = ToolCallLogger("Hospital", seed=3, log_dir=str(run_dir))

        self.assertEqual(logger.log_dir, Path(run_dir))
        self.assertTrue((run_dir / "tool_calls.json").is_file())

    def test_default_layout_unchanged_when_log_dir_omitted(self):
        config = {
            "simulation": {
                "seed": 3,
                "tags": ["mytag"],
                "run_timestamp": "20240101_000000",
            }
        }
        logger = ToolCallLogger("Hospital", seed=3, config=config)

        expected = build_log_dir(
            "Hospital", get_tag_model_subdir(config), 3, "20240101_000000"
        )
        self.assertEqual(logger.log_dir, expected)
        self.assertTrue((expected / "tool_calls.json").is_file())


if __name__ == "__main__":
    unittest.main()
