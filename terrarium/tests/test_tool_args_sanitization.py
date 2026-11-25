import copy
import importlib
import json
import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from terrarium.llm.clients.abstract_client import AbstractClient
from terrarium.llm.clients.tool_args import (
    JSON_REPAIR_WINDOW,
    arguments_would_generate_note,
    close_truncated_json,
    repair_truncated_json,
    response_indicates_length_truncation,
    sanitize_function_call_arguments,
    sanitize_tool_call_arguments,
)


class CloseTruncatedJsonTests(unittest.TestCase):
    def test_complete_json_is_returned_unchanged(self):
        text = '{"a": 1, "b": [1, 2]}'
        self.assertEqual(close_truncated_json(text), text)

    def test_truncated_string_is_closed(self):
        self.assertEqual(close_truncated_json('{"a": "hel'), '{"a": "hel"}')

    def test_nested_braces_and_brackets_closed_in_reverse_order(self):
        self.assertEqual(
            close_truncated_json('{"a": [1, {"b": 2'),
            '{"a": [1, {"b": 2}]}',
        )

    def test_trailing_backslash_before_closing_quote_is_doubled(self):
        prefix = '{"a": "hel' + "\\"
        expected = prefix + "\\" + '"' + "}"
        closed = close_truncated_json(prefix)
        self.assertEqual(closed, expected)
        self.assertEqual(json.loads(closed), {"a": "hel" + "\\"})

    def test_unbalanced_closing_bracket_returns_none(self):
        self.assertIsNone(close_truncated_json("]"))
        self.assertIsNone(close_truncated_json('{"a": 1]'))

    def test_mismatched_closing_bracket_returns_none(self):
        self.assertIsNone(close_truncated_json('{"a": [1}'))


class RepairTruncatedJsonTests(unittest.TestCase):
    def test_valid_json_passes_through_untouched(self):
        text = '{"a": 1, "b": [1, 2]}'
        self.assertEqual(repair_truncated_json(text), text)

    def test_truncated_string_is_repaired(self):
        repaired = repair_truncated_json('{"a": "hel')
        self.assertEqual(repaired, '{"a": "hel"}')
        self.assertEqual(json.loads(repaired), {"a": "hel"})

    def test_truncated_nested_structures_are_repaired(self):
        repaired = repair_truncated_json('{"a": [1, {"b": 2')
        self.assertIsNotNone(repaired)
        self.assertEqual(json.loads(repaired), {"a": [1, {"b": 2}]})

    def test_trailing_backslash_is_repaired(self):
        repaired = repair_truncated_json('{"a": "hel' + "\\")
        self.assertIsNotNone(repaired)
        self.assertEqual(json.loads(repaired), {"a": "hel" + "\\"})

    def test_non_json_prose_returns_none(self):
        self.assertIsNone(repair_truncated_json("hello world, this is not json"))

    def test_non_string_returns_none(self):
        self.assertIsNone(repair_truncated_json(None))
        self.assertIsNone(repair_truncated_json({"a": 1}))

    def test_repair_window_is_bounded(self):
        # The only repairable cut point ('{"a": 1}') sits more than
        # JSON_REPAIR_WINDOW chars from the end, so repair must give up.
        long_tail = '{"a": 1}' + "x" * (JSON_REPAIR_WINDOW + 100)
        self.assertIsNone(repair_truncated_json(long_tail))
        # Control: with a short tail the same cut point is within the window.
        short_tail = '{"a": 1}' + "x" * 10
        self.assertEqual(repair_truncated_json(short_tail), '{"a": 1}')


class SanitizeToolCallArgumentsTests(unittest.TestCase):
    def test_valid_string_arguments_kept_verbatim_with_no_notes(self):
        message = {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "ping", "arguments": '{"payload":"hi"}'},
                }
            ],
        }
        snapshot = copy.deepcopy(message)
        sanitized, notes = sanitize_tool_call_arguments(message)
        self.assertEqual(notes, [])
        self.assertEqual(
            sanitized["tool_calls"][0]["function"]["arguments"], '{"payload":"hi"}'
        )
        self.assertEqual(message, snapshot)

    def test_dict_arguments_are_serialized(self):
        message = {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "ping", "arguments": {"payload": "hi"}},
                }
            ],
        }
        sanitized, notes = sanitize_tool_call_arguments(message)
        self.assertEqual(notes, [])
        arguments = sanitized["tool_calls"][0]["function"]["arguments"]
        self.assertIsInstance(arguments, str)
        self.assertEqual(json.loads(arguments), {"payload": "hi"})
        self.assertEqual(message["tool_calls"][0]["function"]["arguments"], {"payload": "hi"})

    def test_missing_arguments_become_empty_object_with_note(self):
        message = {
            "role": "assistant",
            "tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": "ping"}}
            ],
        }
        sanitized, notes = sanitize_tool_call_arguments(message)
        self.assertEqual(
            sanitized["tool_calls"][0]["function"]["arguments"], "{}"
        )
        self.assertEqual(len(notes), 1)
        self.assertEqual(notes[0]["tool"], "ping")
        self.assertEqual(notes[0]["status"], "missing_arguments")

    def test_truncated_arguments_are_repaired_with_note(self):
        raw = '{"payload": "hel'
        message = {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "ping", "arguments": raw},
                }
            ],
        }
        snapshot = copy.deepcopy(message)
        sanitized, notes = sanitize_tool_call_arguments(message)
        arguments = sanitized["tool_calls"][0]["function"]["arguments"]
        self.assertEqual(json.loads(arguments), {"payload": "hel"})
        self.assertEqual(len(notes), 1)
        self.assertEqual(notes[0]["status"], "repaired")
        self.assertEqual(notes[0]["raw"], raw)
        self.assertEqual(message, snapshot)

    def test_unrepairable_arguments_are_dropped_with_note(self):
        raw = "this is not json at all"
        message = {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "ping", "arguments": raw},
                }
            ],
        }
        sanitized, notes = sanitize_tool_call_arguments(message)
        self.assertEqual(sanitized["tool_calls"][0]["function"]["arguments"], "{}")
        self.assertEqual(len(notes), 1)
        self.assertEqual(notes[0]["status"], "dropped")
        self.assertEqual(notes[0]["raw"], raw)

    def test_every_tool_call_arguments_is_valid_json_string(self):
        message = {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "a", "arguments": '{"x": "trunc'},
                },
                {
                    "id": "call_2",
                    "type": "function",
                    "function": {"name": "b", "arguments": '{"y": 1}'},
                },
                {
                    "id": "call_3",
                    "type": "function",
                    "function": {"name": "c", "arguments": "not json"},
                },
                {
                    "id": "call_4",
                    "type": "function",
                    "function": {"name": "d"},
                },
            ],
        }
        sanitized, notes = sanitize_tool_call_arguments(message)
        for call in sanitized["tool_calls"]:
            arguments = call["function"]["arguments"]
            self.assertIsInstance(arguments, str)
            json.loads(arguments)
        self.assertEqual(
            [note["status"] for note in notes],
            ["repaired", "dropped", "missing_arguments"],
        )

    def test_message_without_tool_calls_is_copied_untouched(self):
        message = {"role": "assistant", "content": "hi"}
        sanitized, notes = sanitize_tool_call_arguments(message)
        self.assertEqual(sanitized, message)
        self.assertIsNot(sanitized, message)
        self.assertEqual(notes, [])


class SanitizeFunctionCallArgumentsTests(unittest.TestCase):
    def test_valid_json_string(self):
        args, note = sanitize_function_call_arguments('{"payload": "hi"}')
        self.assertEqual(args, {"payload": "hi"})
        self.assertIsNone(note)

    def test_dict_passes_through(self):
        args, note = sanitize_function_call_arguments({"payload": "hi"})
        self.assertEqual(args, {"payload": "hi"})
        self.assertIsNone(note)

    def test_none_becomes_empty_dict_with_missing_note(self):
        args, note = sanitize_function_call_arguments(None)
        self.assertEqual(args, {})
        self.assertEqual(note["status"], "missing_arguments")

    def test_truncated_json_is_repaired(self):
        args, note = sanitize_function_call_arguments('{"payload": "hel')
        self.assertEqual(args, {"payload": "hel"})
        self.assertEqual(note["status"], "repaired")

    def test_unrepairable_string_is_dropped(self):
        args, note = sanitize_function_call_arguments("total garbage ]")
        self.assertEqual(args, {})
        self.assertEqual(note["status"], "dropped")


class ArgumentsWouldGenerateNoteTests(unittest.TestCase):
    def test_predicate(self):
        self.assertTrue(arguments_would_generate_note(None))
        self.assertTrue(arguments_would_generate_note('{"a": "trunc'))
        self.assertTrue(arguments_would_generate_note(123))
        self.assertFalse(arguments_would_generate_note({"a": 1}))
        self.assertFalse(arguments_would_generate_note('{"a": 1}'))


class ResponseIndicatesLengthTruncationTests(unittest.TestCase):
    def test_chat_completions_finish_reason_length(self):
        response = {"choices": [{"finish_reason": "length", "message": {}}]}
        self.assertTrue(response_indicates_length_truncation(response))

    def test_chat_completions_other_finish_reason(self):
        response = {"choices": [{"finish_reason": "stop", "message": {}}]}
        self.assertFalse(response_indicates_length_truncation(response))

    def test_responses_api_incomplete_max_output_tokens(self):
        response = SimpleNamespace(
            status="incomplete",
            incomplete_details=SimpleNamespace(reason="max_output_tokens"),
        )
        self.assertTrue(response_indicates_length_truncation(response))

    def test_responses_api_completed(self):
        response = SimpleNamespace(status="completed", incomplete_details=None)
        self.assertFalse(response_indicates_length_truncation(response))

    def test_none_response(self):
        self.assertFalse(response_indicates_length_truncation(None))


class AbstractClientStaticMethodTests(unittest.TestCase):
    def test_staticmethods_delegate_to_shared_helpers(self):
        self.assertEqual(
            AbstractClient.repair_truncated_json('{"a": 1}'), '{"a": 1}'
        )
        self.assertEqual(
            AbstractClient.close_truncated_json('{"a": "hel'), '{"a": "hel"}'
        )
        args, note = AbstractClient.sanitize_function_call_arguments(None)
        self.assertEqual(args, {})
        self.assertEqual(note["status"], "missing_arguments")
        sanitized, notes = AbstractClient.sanitize_tool_call_arguments(
            {"role": "assistant", "tool_calls": []}
        )
        self.assertEqual(sanitized["role"], "assistant")
        self.assertEqual(notes, [])


def _import_fireworks_client_with_mocks():
    fake_requests = types.ModuleType("requests")

    class _Session:
        def post(self, *args, **kwargs):  # pragma: no cover - not used here
            raise AssertionError("network access is not expected in tests")

    fake_requests.Session = _Session
    fake_requests.exceptions = types.SimpleNamespace(RequestException=Exception)

    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda override=True: None

    module_name = "terrarium.llm.clients.fireworks_client"
    if module_name in sys.modules:
        del sys.modules[module_name]

    with patch.dict(
        sys.modules,
        {"requests": fake_requests, "dotenv": fake_dotenv},
    ):
        module = importlib.import_module(module_name)
    return module


def _chat_response(message, finish_reason="stop"):
    return {"choices": [{"finish_reason": finish_reason, "message": message}]}


class FireworksProcessToolCallsTests(unittest.IsolatedAsyncioTestCase):
    def _client(self):
        module = _import_fireworks_client_with_mocks()
        return module.FireworksClient(api_key="test-key")

    async def test_valid_arguments_execute_and_context_stays_valid(self):
        client = self._client()
        message = {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "ext_ping", "arguments": '{"payload":"hi"}'},
                }
            ],
        }
        executed = []

        async def _execute(tool_name, args):
            executed.append((tool_name, args))
            return {"status": "ok"}

        context = []
        tools_executed, updated, step_tools = await client.process_tool_calls(
            _chat_response(message), context, _execute
        )
        self.assertEqual(tools_executed, 1)
        self.assertEqual(executed, [("ext_ping", {"payload": "hi"})])
        self.assertEqual(step_tools, ['ext_ping -- {"payload": "hi"}'])
        self.assertEqual(updated[0]["tool_calls"][0]["function"]["arguments"], '{"payload":"hi"}')
        self.assertEqual(json.loads(updated[1]["content"]), {"status": "ok"})

    async def test_truncated_arguments_are_repaired_and_executed(self):
        client = self._client()
        message = {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "ext_ping", "arguments": '{"payload": "hel'},
                }
            ],
        }
        executed = []

        async def _execute(tool_name, args):
            executed.append((tool_name, args))
            return {"status": "ok"}

        context = []
        with self.assertLogs("terrarium.llm.clients.tool_args", level="WARNING"):
            tools_executed, updated, step_tools = await client.process_tool_calls(
                _chat_response(message, finish_reason="length"), context, _execute
            )
        self.assertEqual(tools_executed, 1)
        self.assertEqual(executed, [("ext_ping", {"payload": "hel"})])
        self.assertIn("ARGS_REPAIRED_FROM_TRUNCATION", step_tools[0])
        # The copy appended to the history must be valid JSON (it is re-POSTed).
        appended_args = updated[0]["tool_calls"][0]["function"]["arguments"]
        self.assertEqual(json.loads(appended_args), {"payload": "hel"})
        self.assertEqual(updated[1]["role"], "tool")
        self.assertEqual(updated[1]["tool_call_id"], "call_1")

    async def test_dropped_arguments_are_not_executed_but_stay_paired(self):
        client = self._client()
        message = {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "ext_ping", "arguments": "not json at all"},
                }
            ],
        }

        async def _execute(tool_name, args):  # pragma: no cover - must not run
            raise AssertionError("dropped tool calls must not execute")

        context = []
        with self.assertLogs("terrarium.llm.clients.tool_args", level="WARNING"):
            tools_executed, updated, step_tools = await client.process_tool_calls(
                _chat_response(message, finish_reason="length"), context, _execute
            )
        self.assertEqual(tools_executed, 0)
        self.assertIn("ARGS_DROPPED_TRUNCATED", step_tools[0])
        # Assistant message in history carries valid (empty) JSON arguments.
        appended_args = updated[0]["tool_calls"][0]["function"]["arguments"]
        self.assertEqual(json.loads(appended_args), {})
        # ... and the call still has a paired tool output with an error payload.
        self.assertEqual(updated[1]["role"], "tool")
        self.assertEqual(updated[1]["tool_call_id"], "call_1")
        output = json.loads(updated[1]["content"])
        self.assertEqual(output["status"], "error")

    async def test_original_response_message_is_not_mutated(self):
        client = self._client()
        raw = '{"payload": "hel'
        message = {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "ext_ping", "arguments": raw},
                }
            ],
        }
        snapshot = copy.deepcopy(message)

        async def _execute(tool_name, args):
            return {"status": "ok"}

        await client.process_tool_calls(_chat_response(message), [], _execute)
        self.assertEqual(message, snapshot)
        self.assertEqual(
            message["tool_calls"][0]["function"]["arguments"], raw
        )


if __name__ == "__main__":
    unittest.main()
