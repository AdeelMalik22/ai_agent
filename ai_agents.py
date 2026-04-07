"""Main entry point for the AI agent system."""

import os
from typing import Any

from openai import OpenAI

from config.settings import load_config
from core.session import initialize_session
from core.streaming import stream_model_response
from guardrails.input_guardrils import (
    GuardrailConfig,
    trim_conversation_history,
    validate_recent_user_repetition,
    validate_user_input,
)
from guardrails.output_guardrils import OutputGuardrailConfig, guard_assistant_output
from system_prompt import AGENT_PROMPTS
from utils.tooling import (
    validate_single_tool_call,
    execute_single_tool,
    extract_tool_names,
    is_valid_response,
    create_tool_error_message,
)


def build_client() -> OpenAI:
    """Build OpenAI/Bedrock client from environment."""
    return OpenAI(
        api_key=os.getenv("OPENAI_API_KEY") or os.getenv("BEDROCK_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL", "https://bedrock-mantle.us-east-1.api.aws/v1"),
    )


def process_tool_calls(
    tool_calls: list[Any],
    messages: list[dict],
    active_agent: str,
    handoffs_this_turn: int,
    known_tools: set[str],
    guardrail_config: GuardrailConfig,
    output_guardrail_config: OutputGuardrailConfig,
    runtime_config: dict,
) -> tuple[str, bool, int]:
    """Process all tool calls from model response.

    Returns: (active_agent, should_continue_loop, updated_handoff_count)
    """
    if runtime_config["debug_handoffs"]:
        tool_names = extract_tool_names(tool_calls)
        print(f"[tools] requested={tool_names}")

    for tool_call in tool_calls:
        is_valid, error_msg, parsed_args = validate_single_tool_call(
            tool_call, known_tools, guardrail_config
        )

        if not is_valid:
            error_message = create_tool_error_message(tool_call, error_msg, output_guardrail_config)
            messages.append(error_message)
            messages = trim_conversation_history(messages, guardrail_config)
            continue

        result, new_agent, handoffs_this_turn = execute_single_tool(
            tool_call,
            parsed_args,
            active_agent,
            handoffs_this_turn,
            runtime_config["max_handoffs_per_turn"],
            output_guardrail_config,
        )

        if new_agent != active_agent:
            active_agent = new_agent
            messages[0] = {"role": "system", "content": AGENT_PROMPTS[active_agent]}
            if runtime_config["debug_handoffs"]:
                print(f"[handoff] -> {active_agent}")

        messages.append({
            "role": "tool",
            "tool_call_id": getattr(tool_call, "id", "unknown"),
            "content": result,
        })
        messages = trim_conversation_history(messages, guardrail_config)

    return active_agent, True, handoffs_this_turn


def handle_assistant_reply(
    collected_content: str,
    messages: list[dict],
    guardrail_config: GuardrailConfig,
    output_guardrail_config: OutputGuardrailConfig,
) -> None:
    """Handle final assistant reply (no tool calls)."""
    reply = guard_assistant_output(collected_content, output_guardrail_config)
    messages.append({"role": "assistant", "content": reply})
    messages = trim_conversation_history(messages, guardrail_config)


def process_model_response(
    collected_content: str,
    tool_calls: list[Any],
    messages: list[dict],
    active_agent: str,
    handoffs_this_turn: int,
    known_tools: set[str],
    guardrail_config: GuardrailConfig,
    output_guardrail_config: OutputGuardrailConfig,
    runtime_config: dict,
) -> tuple[str, int, bool]:
    """Process complete model response (content + tool calls).

    Returns: (active_agent, handoffs_count, should_continue_loop)
    """
    if tool_calls:
        assistant_message_with_tools = {
            "role": "assistant",
            "content": guard_assistant_output(collected_content, output_guardrail_config),
            "tool_calls": [tc.model_dump() for tc in tool_calls if tc is not None],
        }
        messages.append(assistant_message_with_tools)

        active_agent, should_continue, handoffs_this_turn = process_tool_calls(
            tool_calls,
            messages,
            active_agent,
            handoffs_this_turn,
            known_tools,
            guardrail_config,
            output_guardrail_config,
            runtime_config,
        )
        return active_agent, handoffs_this_turn, should_continue
    else:
        handle_assistant_reply(collected_content, messages, guardrail_config, output_guardrail_config)
        return active_agent, handoffs_this_turn, False


def main() -> None:
    """Main chat loop orchestrator."""
    client = build_client()

    guardrail_config, output_guardrail_config, runtime_config = load_config()
    messages, known_tools, active_agent = initialize_session()

    while True:
        try:
            user_input = input("ask Question.....: ").strip()
        except KeyboardInterrupt:
            print("\nBye")
            break
        except EOFError:
            print("Bye")
            break

        if not user_input:
            continue

        if user_input.lower() in {"exit", "quit"}:
            print("Bye")
            break

        user_check = validate_user_input(user_input, guardrail_config)
        if not user_check.allowed:
            print(f"Input rejected: {user_check.reason}")
            continue

        repeat_reason = validate_recent_user_repetition(messages, user_check.normalized_text, guardrail_config)
        if repeat_reason:
            print(f"Input rejected: {repeat_reason}")
            continue

        normalized_text = user_check.normalized_text

        messages.append({"role": "user", "content": normalized_text})
        messages = trim_conversation_history(messages, guardrail_config)
        handoffs_this_turn = 0

        for iteration in range(runtime_config["max_iterations"]):
            try:
                collected_content, tool_calls = stream_model_response(
                    client, messages, runtime_config
                )

                if not is_valid_response(collected_content, tool_calls):
                    print("[WARNING] Empty response from model")
                    continue

                active_agent, handoffs_this_turn, should_continue = process_model_response(
                    collected_content,
                    tool_calls,
                    messages,
                    active_agent,
                    handoffs_this_turn,
                    known_tools,
                    guardrail_config,
                    output_guardrail_config,
                    runtime_config,
                )

                if not should_continue:
                    break

            except Exception as e:
                print(f"[ERROR] Iteration {iteration} failed: {e}")
                if iteration >= runtime_config["max_iterations"] - 1:
                    print("AI: I hit the tool-iteration limit for this request.")
                break

        else:
            print("AI: I hit the tool-iteration limit for this request.")


if __name__ == "__main__":
    main()

