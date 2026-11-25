"""
Shared, provider-agnostic sanitization/repair of truncated LLM tool-call arguments.

When a model hits ``finish_reason=length`` (chat completions) or returns an
incomplete response (Responses API) in the middle of emitting a tool call, the
``function.arguments`` string is cut off mid-token. Left alone, that string
(a) silently parses to ``{}`` when the call is executed and (b) is echoed back
verbatim in the next request's history, which makes the provider reject the
whole request with HTTP 400 ("tool call function.arguments must be valid
JSON").

The helpers here close or repair such truncated JSON without inventing data,
and rewrite assistant messages so that every
``tool_calls[*].function.arguments`` is a string that ``json.loads`` accepts.
They are exposed on :class:`AbstractClient` as thin staticmethods so every
client can call them uniformly.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# How many trailing characters of a truncated ``function.arguments`` string we
# are willing to walk backwards looking for a cut point that yields valid JSON.
# Bounded so a pathological multi-megabyte argument blob cannot turn repair into
# an O(n^2) scan.
JSON_REPAIR_WINDOW = 500


def close_truncated_json(prefix: str) -> Optional[str]:
    """Close any unterminated string/brackets in ``prefix`` (best effort).

    Returns ``prefix`` with a closing quote (if we ended inside a string) and
    the missing closing brackets appended in reverse order, or ``None`` when the
    text contains a structural error that no amount of closing can fix (an
    unbalanced closing bracket, or a bracket that does not match its opener).

    This never invents data: it only terminates what is already open.
    """
    stack: List[str] = []
    in_string = False
    escaped = False
    for ch in prefix:
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            stack.append("{")
        elif ch == "[":
            stack.append("[")
        elif ch in "}]":
            if not stack:
                return None
            opener = stack.pop()
            if (opener == "{" and ch != "}") or (opener == "[" and ch != "]"):
                return None

    out = prefix
    if in_string:
        # A trailing backslash escapes the closing quote we are about to add,
        # so double it first to keep the string terminating.
        if escaped:
            out += "\\"
        out += '"'
    for opener in reversed(stack):
        out += "}" if opener == "{" else "]"
    return out


def repair_truncated_json(text: str) -> Optional[str]:
    """Best-effort repair of a JSON object string that was cut off mid-token.

    Returns a string that ``json.loads`` accepts (valid JSON passes through
    untouched), or ``None`` when the input cannot be repaired.
    """
    if not isinstance(text, str):
        return None

    try:
        json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass
    else:
        # Already valid: hand it back untouched.
        return text

    stripped = text.lstrip()
    if not stripped or stripped[0] not in "{[":
        # Only object/array literals are worth trying to close; a bare scalar or
        # arbitrary prose cannot be repaired without inventing structure.
        return None

    # Walk cut points from the end of the text backwards, bounded to the tail.
    start_cut = max(1, len(text) - JSON_REPAIR_WINDOW)
    for cut in range(len(text), start_cut - 1, -1):
        repaired = close_truncated_json(text[:cut])
        if repaired is None:
            continue
        try:
            json.loads(repaired)
        except (json.JSONDecodeError, ValueError):
            continue
        return repaired
    return None


def arguments_would_generate_note(args: Any) -> bool:
    """Whether an arguments value is one :func:`sanitize_tool_call_arguments`
    rewrites (and therefore attaches a note to)."""
    if args is None:
        return True
    if isinstance(args, dict):
        return False
    if isinstance(args, str):
        try:
            json.loads(args)
        except (json.JSONDecodeError, ValueError):
            return True
        return False
    return True


def sanitize_tool_call_arguments(
    message: Dict[str, Any],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Return ``(sanitized_message, notes)`` for one assistant message.

    Every ``tool_calls[*].function.arguments`` in the returned message is
    guaranteed to be a string that ``json.loads`` accepts. The input is never
    mutated: the returned message is a copy with a new ``tool_calls`` list and a
    new ``function`` dict per call. No keys are added to the message (the
    message is echoed back to the API on the next request, and unknown fields
    can trigger 400s). ``notes`` carries one entry per rewritten call, in call
    order, shaped ``{"tool": name, "status": repaired|dropped|missing_arguments,
    "raw": <truncated raw arguments>}``.
    """
    sanitized: Dict[str, Any] = dict(message)
    notes: List[Dict[str, Any]] = []

    raw_calls = message.get("tool_calls")
    if not raw_calls:
        return sanitized, notes

    new_calls: List[Dict[str, Any]] = []
    for call in raw_calls:
        new_call: Dict[str, Any] = dict(call)
        function_block = dict(call.get("function") or {})
        tool_name = function_block.get("name") or "unknown_tool"
        args = function_block.get("arguments")

        if isinstance(args, dict):
            function_block["arguments"] = json.dumps(args)
        elif args is None:
            function_block["arguments"] = "{}"
            notes.append({"tool": tool_name, "status": "missing_arguments", "raw": ""})
        elif isinstance(args, str):
            try:
                json.loads(args)
            except (json.JSONDecodeError, ValueError):
                repaired = repair_truncated_json(args)
                if repaired is not None:
                    function_block["arguments"] = repaired
                    notes.append(
                        {"tool": tool_name, "status": "repaired", "raw": args[:2000]}
                    )
                else:
                    function_block["arguments"] = "{}"
                    notes.append(
                        {"tool": tool_name, "status": "dropped", "raw": args[:2000]}
                    )
            # A string that already parses is kept verbatim.
        else:
            function_block["arguments"] = "{}"
            notes.append(
                {"tool": tool_name, "status": "dropped", "raw": repr(args)[:2000]}
            )

        new_call["function"] = function_block
        new_calls.append(new_call)

    sanitized["tool_calls"] = new_calls
    return sanitized, notes


def sanitize_function_call_arguments(
    raw: Any,
) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    """Sanitize one Responses-API ``function_call`` item's ``arguments``.

    Returns ``(args_dict, note)`` where ``note`` is ``None`` when ``raw`` was
    already usable as-is (a dict, or a string that parses to a JSON object) and
    otherwise carries ``{"status": repaired|dropped|missing_arguments, "raw":
    ...}``. Truncated JSON that can be closed is repaired; anything else is
    dropped to ``{}`` so the call/output pairing and the next request stay
    valid.
    """
    if raw is None:
        return {}, {"status": "missing_arguments", "raw": ""}
    if isinstance(raw, dict):
        return raw, None
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            repaired = repair_truncated_json(raw)
            if repaired is not None:
                return json.loads(repaired), {"status": "repaired", "raw": raw[:2000]}
            return {}, {"status": "dropped", "raw": raw[:2000]}
        if isinstance(parsed, dict):
            return parsed, None
        # Valid JSON but not an object; tool arguments must be an object.
        return {}, {"status": "dropped", "raw": raw[:2000]}
    return {}, {"status": "dropped", "raw": repr(raw)[:2000]}


def _get_field(obj: Any, field: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(field, default)
    return getattr(obj, field, default)


def response_indicates_length_truncation(response: Any) -> bool:
    """Whether a response object says generation was cut off by a length limit.

    Handles both shapes Terrarium clients see: chat-completions dicts
    (``choices[0].finish_reason == "length"``) and Responses-API objects
    (``status == "incomplete"`` with ``incomplete_details.reason`` of
    ``max_output_tokens``/``length``).
    """
    if response is None:
        return False
    choices = _get_field(response, "choices")
    if choices:
        if _get_field(choices[0], "finish_reason") == "length":
            return True
    if _get_field(response, "status") == "incomplete":
        details = _get_field(response, "incomplete_details")
        if _get_field(details, "reason") in ("max_output_tokens", "length"):
            return True
    return False


def _args_marker(note: Optional[Dict[str, Any]]) -> str:
    """Trajectory marker for a rewritten tool-call arguments string."""
    if not note:
        return ""
    status = note.get("status")
    raw = note.get("raw", "")
    if status == "repaired":
        return f" [ARGS_REPAIRED_FROM_TRUNCATION raw={raw[:300]!r}]"
    if status == "dropped":
        return f" [ARGS_DROPPED_TRUNCATED raw={raw[:300]!r}]"
    return ""


_DROPPED_OUTPUT = json.dumps(
    {
        "status": "error",
        "result": (
            "tool call dropped: the model's arguments were truncated "
            "and could not be repaired"
        ),
    }
)


async def process_chat_tool_calls(
    message: Dict[str, Any],
    context: List[Dict[str, Any]],
    execute_tool_callback: Any,
    *,
    truncated: bool = False,
    log: Optional[logging.Logger] = None,
) -> Tuple[int, List[Dict[str, Any]], List[str]]:
    """Shared chat-completions tool-call handling with truncated-argument repair.

    Sanitizes ``message`` (the assistant message from ``choices[0]``) BEFORE it
    enters the history -- the conversation is re-POSTed verbatim on the next
    step, and a truncated ``function.arguments`` string makes the server reject
    the whole request with HTTP 400 -- then executes each tool call. Repaired
    calls run with the repaired arguments; dropped calls are NOT executed but
    still get an error ``role: tool`` message so the call/output pairing stays
    valid.

    Returns ``(tool_calls_executed, context, step_tools)``.
    """
    effective_log = log or logger
    sanitized, notes = sanitize_tool_call_arguments(message)
    context.append(sanitized)

    notes_iter = iter(notes)
    tool_calls_executed = 0
    step_tools: List[str] = []

    original_calls = message.get("tool_calls") or []
    sanitized_calls = sanitized.get("tool_calls") or []
    # ``notes`` holds one entry per rewritten call, in call order. Pair each
    # note with its call using the same predicate the sanitizer used, so a
    # clean call never steals a later call's note.
    for original_call, call in zip(original_calls, sanitized_calls):
        function_block = call.get("function") or {}
        tool_name = function_block.get("name", "unknown_tool")
        arguments = json.loads(function_block.get("arguments") or "{}")

        original_function = original_call.get("function") or {}
        note = None
        if arguments_would_generate_note(original_function.get("arguments")):
            note = next(notes_iter, None)
        status = note.get("status") if note else None

        step_tools.append(f"{tool_name} -- {json.dumps(arguments)}{_args_marker(note)}")

        if truncated and status in ("repaired", "dropped"):
            effective_log.warning(
                "tool call arguments were truncated by the model "
                "(finish_reason=length) for tool %r; arguments %s",
                tool_name,
                "repaired from the partial JSON"
                if status == "repaired"
                else "dropped (could not be repaired)",
            )

        if status == "dropped":
            # Do not execute, but keep the assistant/tool pairing valid: the
            # API requires a tool message for every tool_call_id.
            context.append(
                {
                    "role": "tool",
                    "tool_call_id": call.get("id"),
                    "name": tool_name,
                    "content": _DROPPED_OUTPUT,
                }
            )
            continue

        result = await execute_tool_callback(tool_name, arguments)
        context.append(
            {
                "role": "tool",
                "tool_call_id": call.get("id"),
                "name": tool_name,
                "content": json.dumps(result),
            }
        )
        tool_calls_executed += 1

    return tool_calls_executed, context, step_tools
