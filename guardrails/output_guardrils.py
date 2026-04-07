"""Output guardrails for assistant replies and tool outputs."""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any


DEFAULT_MAX_ASSISTANT_OUTPUT_LENGTH = 6000
DEFAULT_MAX_TOOL_OUTPUT_LENGTH = 3000
DEFAULT_EMPTY_ASSISTANT_FALLBACK = "I cannot provide that response safely. Please rephrase your request."
DEFAULT_EMPTY_TOOL_FALLBACK = '{"error":"Tool output was blocked by output guardrails"}'
DEFAULT_HATEFUL_ASSISTANT_FALLBACK = "I cannot help with hateful or abusive content."
DEFAULT_HATEFUL_TOOL_FALLBACK = '{"error":"Tool output contained hateful content and was blocked"}'

_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

_SECRET_PATTERNS = (
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bASIA[0-9A-Z]{16}\b"),
    re.compile(r"(?i)(authorization\s*:\s*bearer\s+)[A-Za-z0-9._\-]+"),
    re.compile(r"(?i)\b(api[_-]?key|token|secret|password)\s*[=:]\s*[^\s,;\"]+"),
    re.compile(r"\bsk-[A-Za-z0-9]{20,}\b"),
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
class OutputGuardrailConfig:
    max_assistant_output_length: int = DEFAULT_MAX_ASSISTANT_OUTPUT_LENGTH
    max_tool_output_length: int = DEFAULT_MAX_TOOL_OUTPUT_LENGTH
    empty_assistant_fallback: str = DEFAULT_EMPTY_ASSISTANT_FALLBACK
    empty_tool_fallback: str = DEFAULT_EMPTY_TOOL_FALLBACK
    block_hateful_output: bool = True
    hateful_assistant_fallback: str = DEFAULT_HATEFUL_ASSISTANT_FALLBACK
    hateful_tool_fallback: str = DEFAULT_HATEFUL_TOOL_FALLBACK


def _strip_control_chars(text: str) -> str:
    return _CONTROL_CHARS_RE.sub("", text)


def _redact_secrets(text: str) -> str:
    redacted = text
    for pattern in _SECRET_PATTERNS:
        redacted = pattern.sub("[REDACTED]", redacted)
    return redacted


def _sanitize_text(text: str) -> str:
    return _redact_secrets(_strip_control_chars(text)).strip()


def _contains_hateful_speech(text: str) -> bool:
    for pattern in _HATEFUL_PATTERNS:
        if pattern.search(text):
            return True
    return False


def _sanitize_json_value(value: Any) -> Any:
    if isinstance(value, str):
        sanitized = _sanitize_text(value)
        if _contains_hateful_speech(sanitized):
            return "[BLOCKED_HATEFUL_CONTENT]"
        return sanitized
    if isinstance(value, dict):
        return {str(k): _sanitize_json_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize_json_value(item) for item in value]
    return value


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "\n...[truncated]"


def guard_assistant_output(content: str | None, config: OutputGuardrailConfig | None = None) -> str:
    config = config or OutputGuardrailConfig()
    sanitized = _sanitize_text(content or "")
    if config.block_hateful_output and _contains_hateful_speech(sanitized):
        return config.hateful_assistant_fallback
    truncated = _truncate(sanitized, config.max_assistant_output_length)
    return truncated or config.empty_assistant_fallback


def guard_tool_output(content: str | None, config: OutputGuardrailConfig | None = None) -> str:
    config = config or OutputGuardrailConfig()
    raw = content or ""

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        sanitized = _truncate(_sanitize_text(raw), config.max_tool_output_length)
        if config.block_hateful_output and _contains_hateful_speech(sanitized):
            return config.hateful_tool_fallback
        return sanitized or config.empty_tool_fallback

    sanitized_json = _sanitize_json_value(parsed)
    serialized = json.dumps(sanitized_json)
    if config.block_hateful_output and _contains_hateful_speech(serialized):
        return config.hateful_tool_fallback
    if len(serialized) > config.max_tool_output_length:
        preview = serialized[: config.max_tool_output_length]
        return json.dumps({"warning": "Tool output truncated by guardrails", "preview": preview})
    return serialized or config.empty_tool_fallback

