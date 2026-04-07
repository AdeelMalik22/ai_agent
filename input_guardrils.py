"""Input guardrails for user prompts and model/tool payloads.

This module keeps validation rules in one place so the chat loop can reject
unsafe or oversized inputs before they reach the model or tool layer.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any


DEFAULT_MAX_USER_INPUT_LENGTH = 4000
DEFAULT_MAX_TOOL_ARGUMENTS_LENGTH = 4000
DEFAULT_MAX_HANDOFF_REASON_LENGTH = 300
DEFAULT_MAX_HISTORY_MESSAGES = 40
DEFAULT_REPEAT_MESSAGE_WINDOW = 6
DEFAULT_REPEAT_MESSAGE_MAX_COUNT = 2
DEFAULT_MAX_JSON_KEYS = 50
DEFAULT_MAX_JSON_DEPTH = 5
DEFAULT_MAX_JSON_LIST_ITEMS = 50
DEFAULT_BLOCK_HATEFUL_INPUT = True

_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_WHITESPACE_RE = re.compile(r"\s+")
_REPEATED_CHAR_RE = re.compile(r"(.)\1{30,}")

_BLOCKED_PHRASES = (
    "ignore previous instructions",
    "reveal your system prompt",
    "show hidden prompt",
    "bypass safety",
    "jailbreak",
)

_PROTECTED_GROUPS = (
    "race",
    "religion",
    "ethnicity",
    "nationality",
    "immigrant",
    "refugee",
    "black",
    "white",
    "asian",
    "latino",
    "muslim",
    "christian",
    "jewish",
    "hindu",
    "gay",
    "lesbian",
    "trans",
    "woman",
    "women",
    "man",
    "men",
    "disabled",
)
_GROUP_PATTERN = "|".join(_PROTECTED_GROUPS)

_HATEFUL_PATTERNS = (
    re.compile(rf"\b(i\s+)?hate\s+(all\s+)?({_GROUP_PATTERN})s?\b", re.IGNORECASE),
    re.compile(
        rf"\b({_GROUP_PATTERN})s?\b.{{0,30}}\b(are|is)\b.{{0,30}}\b(inferior|vermin|animals|dirty|disease|subhuman)\b",
        re.IGNORECASE,
    ),
    re.compile(
        rf"\b(ban|remove|eliminate|deport|kill|attack|hurt)\s+(all\s+)?({_GROUP_PATTERN})s?\b",
        re.IGNORECASE,
    ),
)


@dataclass(frozen=True)
class GuardrailResult:
    allowed: bool
    normalized_text: str
    reason: str = ""


@dataclass(frozen=True)
class GuardrailConfig:
    max_user_input_length: int = DEFAULT_MAX_USER_INPUT_LENGTH
    max_tool_arguments_length: int = DEFAULT_MAX_TOOL_ARGUMENTS_LENGTH
    max_handoff_reason_length: int = DEFAULT_MAX_HANDOFF_REASON_LENGTH
    max_history_messages: int = DEFAULT_MAX_HISTORY_MESSAGES
    repeat_message_window: int = DEFAULT_REPEAT_MESSAGE_WINDOW
    repeat_message_max_count: int = DEFAULT_REPEAT_MESSAGE_MAX_COUNT
    max_json_keys: int = DEFAULT_MAX_JSON_KEYS
    max_json_depth: int = DEFAULT_MAX_JSON_DEPTH
    max_json_list_items: int = DEFAULT_MAX_JSON_LIST_ITEMS
    block_hateful_input: bool = DEFAULT_BLOCK_HATEFUL_INPUT


def normalize_text(text: str) -> str:
    """Collapse repeated whitespace and trim outer spaces."""
    return _WHITESPACE_RE.sub(" ", text).strip()


def validate_user_input(user_input: str, config: GuardrailConfig | None = None) -> GuardrailResult:
    """Validate raw user input before sending it to the model."""
    strict_config = config if config is not None else GuardrailConfig()

    if user_input is None:
        return GuardrailResult(False, "", "Input cannot be empty")

    normalized = normalize_text(user_input)
    if not normalized:
        return GuardrailResult(False, "", "Input cannot be empty")

    if len(normalized) > strict_config.max_user_input_length:
        return GuardrailResult(
            False,
            normalized[: strict_config.max_user_input_length],
            f"Input exceeds the maximum length of {strict_config.max_user_input_length} characters",
        )

    if _CONTROL_CHARS_RE.search(normalized):
        return GuardrailResult(False, "", "Input contains unsupported control characters")

    if _REPEATED_CHAR_RE.search(normalized):
        return GuardrailResult(False, "", "Input looks like repeated-character spam")

    lowered = normalized.lower()
    for phrase in _BLOCKED_PHRASES:
        if phrase in lowered:
            return GuardrailResult(False, "", "Input contains blocked prompt-injection text")

    if strict_config.block_hateful_input and contains_hateful_speech(normalized):
        return GuardrailResult(False, "", "Input contains hateful speech")

    return GuardrailResult(True, normalized)


def contains_hateful_speech(text: str) -> bool:
    """Detect broad hate-speech patterns without maintaining explicit slur lists."""
    for pattern in _HATEFUL_PATTERNS:
        if pattern.search(text):
            return True
    return False


def _validate_json_shape(value: Any, *, depth: int, config: GuardrailConfig) -> str | None:
    if depth > config.max_json_depth:
        return f"Arguments nesting exceeds max depth of {config.max_json_depth}"

    if isinstance(value, dict):
        if len(value) > config.max_json_keys:
            return f"Arguments object exceeds max key count of {config.max_json_keys}"
        for nested in value.values():
            error = _validate_json_shape(nested, depth=depth + 1, config=config)
            if error:
                return error
        return None

    if isinstance(value, list):
        if len(value) > config.max_json_list_items:
            return f"Arguments list exceeds max item count of {config.max_json_list_items}"
        for item in value:
            error = _validate_json_shape(item, depth=depth + 1, config=config)
            if error:
                return error
        return None

    return None


def validate_json_arguments(
    raw_arguments: str,
    *,
    max_length: int | None = None,
    config: GuardrailConfig | None = None,
) -> tuple[dict[str, Any], str | None]:
    """Parse and validate JSON arguments used by tool calls or handoffs.

    Returns:
        A tuple of (parsed_args, error_message). If validation succeeds, the
        error message is None.
    """
    strict_config = config if config is not None else GuardrailConfig()
    limit = max_length or DEFAULT_MAX_TOOL_ARGUMENTS_LENGTH

    if raw_arguments is None:
        return {}, None

    if len(raw_arguments) > limit:
        return {}, f"Arguments exceed the maximum length of {limit} characters"

    try:
        parsed = json.loads(raw_arguments) if raw_arguments else {}
    except json.JSONDecodeError as exc:
        return {}, f"Invalid JSON arguments: {exc}"

    if not isinstance(parsed, dict):
        return {}, "Tool arguments must be a JSON object"

    shape_error = _validate_json_shape(parsed, depth=1, config=strict_config)
    if shape_error:
        return {}, shape_error

    return parsed, None


def validate_handoff_reason(reason: str | None, config: GuardrailConfig | None = None) -> tuple[str, str | None]:
    """Normalize and cap the handoff reason before it is stored or logged."""
    config = config or GuardrailConfig()
    reason_text = normalize_text(reason or "")
    if len(reason_text) > config.max_handoff_reason_length:
        return reason_text[: config.max_handoff_reason_length], (
            f"Handoff reason exceeds the maximum length of {config.max_handoff_reason_length} characters"
        )
    return reason_text, None


def is_known_tool(tool_name: str, allowed_tools: set[str]) -> bool:
    return tool_name in allowed_tools


def validate_recent_user_repetition(
    messages: list[dict[str, Any]],
    new_input: str,
    config: GuardrailConfig | None = None,
) -> str | None:
    """Reject repeated user inputs that look like accidental resend loops."""
    config = config or GuardrailConfig()
    repeated = 0
    remaining = config.repeat_message_window

    for message in reversed(messages):
        if remaining <= 0:
            break
        if message.get("role") == "user":
            remaining -= 1
            if normalize_text(str(message.get("content", ""))) == new_input:
                repeated += 1
    if repeated >= config.repeat_message_max_count:
        return "Input repeated too many times in recent history"
    return None


def trim_conversation_history(messages: list[dict[str, Any]], config: GuardrailConfig | None = None) -> list[dict[str, Any]]:
    """Keep context bounded while preserving the system message and recent turns."""
    config = config or GuardrailConfig()
    if len(messages) <= config.max_history_messages:
        return messages

    system_message = messages[0] if messages and messages[0].get("role") == "system" else None
    tail_size = config.max_history_messages - 1 if system_message else config.max_history_messages
    trimmed_tail = messages[-tail_size:]
    if system_message:
        return [system_message] + trimmed_tail
    return trimmed_tail


