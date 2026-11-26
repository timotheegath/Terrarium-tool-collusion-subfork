"""Branch-local tests for vibe/blackboard-logger-log-dir.

Pins the new optional ``log_dir`` parameter of ``BlackboardLogger`` and the
config-based log root resolution:

- when ``log_dir`` is passed (as a ``Path`` or ``str``), blackboard artifacts
  are written directly into that directory, so runners can point the logger at
  an arbitrary run directory instead of the forced
  <root>/<env>/<tag_model>/<timestamp>/seed_<seed> layout;
- when ``log_dir`` is omitted, the log root is resolved from the
  ``logging.root_dir`` config key via ``get_log_root_dir`` (default ``logs/``),
  matching what ``clear_seed_directories`` clears;
- ``SequentialCommunicationProtocol`` accepts an injected ``blackboard_logger``
  and otherwise constructs one inline as before.

Runs without any LLM/API access.

Run from the repo root with the local package shadowing any installed copy:

    PYTHONPATH=$PWD <python> -m unittest terrarium.tests.test_blackboard_logger_log_dir -v
    # or, if pytest is available:
    PYTHONPATH=$PWD <python> -m pytest terrarium/tests/test_blackboard_logger_log_dir.py -v
"""

import os
import tempfile
import unittest
from pathlib import Path

from terrarium.communication_protocols.sequential import SequentialCommunicationProtocol
from terrarium.core.logger import BlackboardLogger
from terrarium.utils import build_log_dir, get_log_root_dir, get_tag_model_subdir


def _config(**extra):
    config = {
        "environment": {"name": "Hospital"},
        "simulation": {
            "seed": 3,
            "tags": ["mytag"],
            "run_timestamp": "20240101_000000",
        },
    }
    config.update(extra)
    return config


class BlackboardLoggerLogDirTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        self._cwd = os.getcwd()
        os.chdir(self.tmp)
        self.addCleanup(os.chdir, self._cwd)

    def test_explicit_log_dir_path_is_used(self):
        run_dir = self.tmp / "run_42"
        logger = BlackboardLogger(_config(), log_dir=run_dir)

        self.assertEqual(logger.log_root, run_dir)

        logger.log_blackboard_creation("1", ["agent_a", "agent_b"], inviter="agent_a")
        log_file = run_dir / "blackboard_1.txt"
        self.assertTrue(log_file.is_file())
        self.assertIn("BLACKBOARD CREATED", log_file.read_text(encoding="utf-8"))

    def test_explicit_log_dir_accepts_str(self):
        run_dir = self.tmp / "run_str"
        logger = BlackboardLogger(_config(), log_dir=str(run_dir))

        self.assertEqual(logger.log_root, Path(run_dir))
        self.assertTrue(run_dir.is_dir())

    def test_logging_root_dir_config_key_is_honored(self):
        custom_root = self.tmp / "experiments" / "proj" / "logs"
        config = _config(logging={"root_dir": str(custom_root)})
        logger = BlackboardLogger(config)

        expected = build_log_dir(
            "Hospital",
            get_tag_model_subdir(config),
            3,
            "20240101_000000",
            log_root=get_log_root_dir(config),
        )
        self.assertEqual(logger.log_root, expected)
        self.assertEqual(logger.log_root.parts[: len(custom_root.parts)], custom_root.parts)
        self.assertTrue(logger.log_root.is_dir())

    def test_default_layout_unchanged_when_log_dir_and_logging_config_omitted(self):
        config = _config()
        logger = BlackboardLogger(config)

        expected = build_log_dir(
            "Hospital", get_tag_model_subdir(config), 3, "20240101_000000"
        )
        self.assertEqual(logger.log_root, expected)
        self.assertEqual(logger.log_root.parts[0], "logs")


class SequentialCommunicationProtocolBlackboardLoggerTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        self._cwd = os.getcwd()
        os.chdir(self.tmp)
        self.addCleanup(os.chdir, self._cwd)

    def test_injected_blackboard_logger_is_used(self):
        run_dir = self.tmp / "run_7"
        injected = BlackboardLogger(_config(), log_dir=run_dir)
        protocol = SequentialCommunicationProtocol(
            _config(), tool_logger=None, blackboard_logger=injected
        )

        self.assertIs(protocol.blackboard_logger, injected)
        self.assertEqual(protocol.blackboard_logger.log_root, run_dir)

    def test_default_inline_blackboard_logger_unchanged(self):
        config = _config(logging={"root_dir": str(self.tmp / "custom_logs")})
        protocol = SequentialCommunicationProtocol(config, tool_logger=None)

        self.assertIsInstance(protocol.blackboard_logger, BlackboardLogger)
        expected = build_log_dir(
            "Hospital",
            get_tag_model_subdir(config),
            3,
            "20240101_000000",
            log_root=get_log_root_dir(config),
        )
        self.assertEqual(protocol.blackboard_logger.log_root, expected)


if __name__ == "__main__":
    unittest.main()
