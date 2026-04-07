"""Configuration loader for runtime and guardrail settings."""

from __future__ import annotations

import os
from typing import Any

from guardrails.input_guardrils import GuardrailConfig
from guardrails.output_guardrils import OutputGuardrailConfig


def load_config() -> tuple[GuardrailConfig, OutputGuardrailConfig, dict[str, Any]]:
    guardrail_config = GuardrailConfig(
        max_user_input_length=int(os.getenv("MAX_USER_INPUT_LENGTH", "4000")),
        max_tool_arguments_length=int(os.getenv("MAX_TOOL_ARGUMENTS_LENGTH", "4000")),
        max_handoff_reason_length=int(os.getenv("MAX_HANDOFF_REASON_LENGTH", "300")),
        max_history_messages=int(os.getenv("MAX_HISTORY_MESSAGES", "40")),
        repeat_message_window=int(os.getenv("REPEAT_MESSAGE_WINDOW", "6")),
        repeat_message_max_count=int(os.getenv("REPEAT_MESSAGE_MAX_COUNT", "2")),
        max_json_keys=int(os.getenv("MAX_JSON_KEYS", "50")),
        max_json_depth=int(os.getenv("MAX_JSON_DEPTH", "5")),
        max_json_list_items=int(os.getenv("MAX_JSON_LIST_ITEMS", "50")),
        block_hateful_input=os.getenv("BLOCK_HATEFUL_INPUT", "1") == "1",
    )

    output_guardrail_config = OutputGuardrailConfig(
        max_assistant_output_length=int(os.getenv("MAX_ASSISTANT_OUTPUT_LENGTH", "6000")),
        max_tool_output_length=int(os.getenv("MAX_TOOL_OUTPUT_LENGTH", "3000")),
        empty_assistant_fallback=os.getenv(
            "EMPTY_ASSISTANT_FALLBACK",
            "I cannot provide that response safely. Please rephrase your request.",
        ),
        empty_tool_fallback=os.getenv(
            "EMPTY_TOOL_FALLBACK",
            '{"error":"Tool output was blocked by output guardrails"}',
        ),
        block_hateful_output=os.getenv("BLOCK_HATEFUL_OUTPUT", "1") == "1",
        hateful_assistant_fallback=os.getenv(
            "HATEFUL_ASSISTANT_FALLBACK",
            "I cannot help with hateful or abusive content.",
        ),
        hateful_tool_fallback=os.getenv(
            "HATEFUL_TOOL_FALLBACK",
            '{"error":"Tool output contained hateful content and was blocked"}',
        ),
    )

    runtime_config = {
        "model": os.getenv("MODEL_ID", "openai.gpt-oss-120b"),
        "max_iterations": int(os.getenv("MAX_TOOL_ITERATIONS", "8")),
        "max_handoffs_per_turn": int(os.getenv("MAX_HANDOFFS_PER_TURN", "2")),
        "debug_handoffs": os.getenv("DEBUG_HANDOFFS", "1") == "1",
        "stream_delay": float(os.getenv("STREAM_DELAY", "0.05")),
    }

    return guardrail_config, output_guardrail_config, runtime_config

