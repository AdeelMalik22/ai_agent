import os
import time
from typing import Any
from openai import OpenAI

from input_guardrils import (
    GuardrailConfig,
    is_known_tool,
    trim_conversation_history,
    validate_json_arguments,
    validate_recent_user_repetition,
    validate_user_input,
)
from output_guardrils import OutputGuardrailConfig, guard_assistant_output, guard_tool_output
from system_prompt import AGENT_PROMPTS
from tools import TOOLS, run_tool, HANDOFF_TOOL, execute_handoff


def build_client() -> OpenAI:
    return OpenAI(
        api_key=os.getenv("OPENAI_API_KEY") or os.getenv("BEDROCK_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL", "https://bedrock-mantle.us-east-1.api.aws/v1"),
    )


def load_config() -> tuple[GuardrailConfig, OutputGuardrailConfig, dict[str, Any]]:
    """Load all configurations from environment variables."""
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


def initialize_session(runtime_config: dict) -> tuple[list[dict], set[str], str]:
    """Initialize message history, known tools, and active agent."""
    active_agent = os.getenv("DEFAULT_AGENT", "general")
    if active_agent not in AGENT_PROMPTS:
        active_agent = "general"

    messages: list[dict] = [{"role": "system", "content": AGENT_PROMPTS[active_agent]}]

    known_tools = {tool["function"]["name"] for tool in TOOLS}
    known_tools.add(HANDOFF_TOOL["function"]["name"])

    return messages, known_tools, active_agent


def validate_and_normalize_input(
    user_input: str,
    messages: list[dict],
    guardrail_config: GuardrailConfig,
) -> tuple[bool, str, str]:
    """Validate user input through all guardrails.

    Returns: (is_valid, normalized_text, reason_if_invalid)
    """
    user_check = validate_user_input(user_input, guardrail_config)
    if not user_check.allowed:
        return False, "", user_check.reason

    repeat_reason = validate_recent_user_repetition(messages, user_check.normalized_text, guardrail_config)
    if repeat_reason:
        return False, "", repeat_reason

    return True, user_check.normalized_text, ""


def stream_model_response(
    client: OpenAI,
    messages: list[dict],
    runtime_config: dict,
    output_guardrail_config: OutputGuardrailConfig,
) -> tuple[str, list[Any]]:
    """Stream response from model and collect text + tool calls.

    Returns: (collected_content, tool_calls_list)
    """
    response = client.chat.completions.create(
        model=runtime_config["model"],
        messages=messages,
        tools=TOOLS + [HANDOFF_TOOL],
        tool_choice="auto",
        stream=True,
    )  # type: ignore

    collected_content = ""
    tool_calls = []

    print("AI: ", end="", flush=True)

    try:
        for chunk in response:
            if not chunk.choices or not chunk.choices[0]:
                continue

            delta = chunk.choices[0].delta

            if hasattr(delta, "content") and delta.content:
                content_chunk = delta.content
                collected_content += content_chunk
                print(content_chunk, end="", flush=True)
                time.sleep(runtime_config["stream_delay"])

            if hasattr(delta, "tool_calls") and delta.tool_calls:
                valid_calls = [tc for tc in delta.tool_calls if tc is not None]
                tool_calls.extend(valid_calls)

    except Exception as e:
        print(f"\n[ERROR] Streaming failed: {e}")
        return collected_content, []

    print()
    return collected_content, tool_calls


def validate_single_tool_call(
    tool_call: Any,
    known_tools: set[str],
    guardrail_config: GuardrailConfig,
) -> tuple[bool, str, dict[str, Any]]:
    """Validate a single tool call.

    Returns: (is_valid, error_message, parsed_args)
    """
    if not hasattr(tool_call, "function"):
        return False, "Tool call missing function attribute", {}

    tool_name = getattr(tool_call.function, "name", None)
    if not tool_name:
        return False, "Tool call missing function name", {}
    
    if not isinstance(tool_name, str):
        return False, "Tool name is not a string", {}

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
    runtime_config: dict,
    output_guardrail_config: OutputGuardrailConfig,
) -> tuple[str, str, int]:
    """Execute a single tool call and return (result_json, new_agent, new_handoff_count).

    Returns: (result_string, updated_agent, updated_handoff_count)
    """
    from json import dumps as json_dumps

    tool_name = tool_call.function.name

    if tool_name == "handoff_to_agent":
        result, new_agent, updated_count = execute_handoff(
            raw_arguments=json_dumps(parsed_args),
            active_agent=active_agent,
            handoffs_this_turn=handoffs_this_turn,
            max_handoffs=max_handoffs,
        )
        return result, new_agent, updated_count
    else:
        try:
            result = run_tool(
                tool_name=tool_name,
                raw_arguments=json_dumps(parsed_args),
            )
            guarded_result = guard_tool_output(result, output_guardrail_config)
            return guarded_result, active_agent, handoffs_this_turn
        except Exception as e:
            error_result = json_dumps({"error": f"Tool execution failed: {str(e)}"})
            guarded_error = guard_tool_output(error_result, output_guardrail_config)
            return guarded_error, active_agent, handoffs_this_turn


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
    from json import dumps as json_dumps

    if runtime_config["debug_handoffs"]:
        tool_names = [tc.function.name for tc in tool_calls if hasattr(tc, "function")]
        print(f"[tools] requested={tool_names}")

    for tool_call in tool_calls:
        is_valid, error_msg, parsed_args = validate_single_tool_call(
            tool_call, known_tools, guardrail_config
        )

        if not is_valid:
            error_json = json_dumps({"error": error_msg})
            guarded_error = guard_tool_output(error_json, output_guardrail_config)
            messages.append({
                "role": "tool",
                "tool_call_id": getattr(tool_call, "id", "unknown"),
                "content": guarded_error,
            })
            messages = trim_conversation_history(messages, guardrail_config)
            continue

        result, new_agent, handoffs_this_turn = execute_single_tool(
            tool_call,
            parsed_args,
            active_agent,
            handoffs_this_turn,
            runtime_config["max_handoffs_per_turn"],
            runtime_config,
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
) -> bool:
    """Handle final assistant reply (no tool calls).

    Returns: should_break_loop (True to exit tool loop)
    """
    reply = guard_assistant_output(collected_content, output_guardrail_config)
    messages.append({"role": "assistant", "content": reply})
    messages = trim_conversation_history(messages, guardrail_config)
    return True


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
    assistant_message_with_tools: dict[str, Any] | None = None,
) -> tuple[str, int, bool]:
    """Process complete model response (content + tool calls).

    Returns: (active_agent, handoffs_count, should_continue_loop)
    """
    if tool_calls:
        if assistant_message_with_tools:
            guarded_content = guard_assistant_output(collected_content, output_guardrail_config)
            assistant_message_with_tools["content"] = guarded_content
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
    messages, known_tools, active_agent = initialize_session(runtime_config)

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

        is_valid, normalized_text, rejection_reason = validate_and_normalize_input(
            user_input, messages, guardrail_config
        )

        if not is_valid:
            print(f"Input rejected: {rejection_reason}")
            continue

        messages.append({"role": "user", "content": normalized_text})
        messages = trim_conversation_history(messages, guardrail_config)
        handoffs_this_turn = 0

        for iteration in range(runtime_config["max_iterations"]):
            try:
                collected_content, tool_calls = stream_model_response(
                    client, messages, runtime_config, output_guardrail_config
                )

                if not collected_content and not tool_calls:
                    print("[WARNING] Empty response from model")
                    continue

                assistant_message_with_tools = None
                if tool_calls:
                    assistant_message_with_tools = {
                        "role": "assistant",
                        "content": collected_content,
                        "tool_calls": [tc.model_dump() for tc in tool_calls if tc is not None],
                    }

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
                    assistant_message_with_tools,
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
