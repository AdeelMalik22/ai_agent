"""Tool validation and execution helpers."""

from __future__ import annotations

from json import dumps as json_dumps
from typing import Any

from guardrails.input_guardrils import GuardrailConfig, is_known_tool, validate_json_arguments
from guardrails.output_guardrils import OutputGuardrailConfig, guard_tool_output
from tools import execute_handoff, run_tool


def validate_single_tool_call(
    tool_call: Any,
    known_tools: set[str],
    guardrail_config: GuardrailConfig,
) -> tuple[bool, str, dict[str, Any]]:
    if not hasattr(tool_call, "function"):
        return False, "Tool call missing function attribute", {}

    tool_name = getattr(tool_call.function, "name", None)
    if not isinstance(tool_name, str) or not tool_name:
        return False, "Tool call missing function name", {}

    if not is_known_tool(tool_name, known_tools):
        return False, f"Unknown tool requested: {tool_name}", {}

    raw_args = getattr(tool_call.function, "arguments", "")
    parsed_args, arg_error = validate_json_arguments(
        raw_args,
        max_length=guardrail_config.max_tool_arguments_length,
        config=guardrail_config,
    )
    if arg_error:
        return False, arg_error, {}

    return True, "", parsed_args


def execute_single_tool(
    tool_call: Any,
    parsed_args: dict[str, Any],
    active_agent: str,
    handoffs_this_turn: int,
    max_handoffs: int,
    output_guardrail_config: OutputGuardrailConfig,
) -> tuple[str, str, int]:
    tool_name = tool_call.function.name
    if tool_name == "handoff_to_agent":
        return execute_handoff(
            raw_arguments=json_dumps(parsed_args),
            active_agent=active_agent,
            handoffs_this_turn=handoffs_this_turn,
            max_handoffs=max_handoffs,
        )

    result = run_tool(tool_name=tool_name, raw_arguments=json_dumps(parsed_args))
    return guard_tool_output(result, output_guardrail_config), active_agent, handoffs_this_turn


def extract_tool_names(tool_calls: list[Any]) -> list[str]:
    return [getattr(tc.function, "name", "unknown") for tc in tool_calls if hasattr(tc, "function")]


def create_tool_error_message(tool_call: Any, error_msg: str, output_cfg: OutputGuardrailConfig) -> dict[str, Any]:
    return {
        "role": "tool",
        "tool_call_id": getattr(tool_call, "id", "unknown"),
        "content": guard_tool_output(json_dumps({"error": error_msg}), output_cfg),
    }


def is_valid_response(collected_content: str, tool_calls: list[Any]) -> bool:
    return bool(collected_content or tool_calls)

