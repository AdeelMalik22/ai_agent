import os
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
from system_prompt import AGENT_PROMPTS
from tools import TOOLS, run_tool, HANDOFF_TOOL, execute_handoff


def build_client() -> OpenAI:
    return OpenAI(
        api_key=os.getenv("OPENAI_API_KEY") or os.getenv("BEDROCK_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL", "https://bedrock-mantle.us-east-1.api.aws/v1"),
    )


def main() -> None:
    client = build_client()
    from json import dumps as json_dumps

    model = os.getenv("MODEL_ID", "openai.gpt-oss-120b")
    max_iterations = int(os.getenv("MAX_TOOL_ITERATIONS", "8"))
    max_handoffs_per_turn = int(os.getenv("MAX_HANDOFFS_PER_TURN", "2"))
    debug_handoffs = os.getenv("DEBUG_HANDOFFS", "1") == "1"
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
    )
    active_agent = os.getenv("DEFAULT_AGENT", "general")
    if active_agent not in AGENT_PROMPTS:
        active_agent = "general"
    messages: Any = [{"role": "system", "content": AGENT_PROMPTS[active_agent]}]
    known_tools = {tool["function"]["name"] for tool in TOOLS}
    known_tools.add(HANDOFF_TOOL["function"]["name"])

    while True:
        user_input = input("ask Question.....: ").strip()
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

        messages.append({"role": "user", "content": user_check.normalized_text})
        messages = trim_conversation_history(messages, guardrail_config)
        handoffs_this_turn = 0

        for _ in range(max_iterations):
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=TOOLS + [HANDOFF_TOOL],
                tool_choice="auto",
            )

            message = response.choices[0].message
            tool_calls = message.tool_calls or []
            if debug_handoffs and tool_calls:
                print(f"[tools] requested={[call.function.name for call in tool_calls]}")

            if tool_calls:
                assistant_message = {
                    "role": "assistant",
                    "content": message.content or "",
                    "tool_calls": [tc.model_dump() for tc in tool_calls],
                }
                messages.append(assistant_message)

                for tool_call in tool_calls:
                    if not is_known_tool(tool_call.function.name, known_tools):
                        result = json_dumps({"error": f"Blocked unknown tool: {tool_call.function.name}"})
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": result,
                            }
                        )
                        continue

                    parsed_args, arg_error = validate_json_arguments(
                        tool_call.function.arguments,
                        max_length=guardrail_config.max_tool_arguments_length,
                        config=guardrail_config,
                    )
                    if arg_error:
                        result = json_dumps({"error": arg_error})
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": result,
                            }
                        )
                        continue

                    if tool_call.function.name == "handoff_to_agent":
                        old_agent = active_agent
                        result, new_agent, handoffs_this_turn = execute_handoff(
                            raw_arguments=json_dumps(parsed_args),
                            active_agent=active_agent,
                            handoffs_this_turn=handoffs_this_turn,
                            max_handoffs=max_handoffs_per_turn,
                        )
                        if new_agent != active_agent:
                            active_agent = new_agent
                            # Update active system prompt after a successful handoff.
                            messages[0] = {"role": "system", "content": AGENT_PROMPTS[active_agent]}
                            if debug_handoffs:
                                print(f"[handoff] {old_agent} -> {active_agent}")
                        elif debug_handoffs:
                            print(f"[handoff] no change ({old_agent})")
                    else:
                        result = run_tool(
                            tool_name=tool_call.function.name,
                            raw_arguments=json_dumps(parsed_args),
                        )
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": result,
                        }
                    )
                    messages = trim_conversation_history(messages, guardrail_config)
                continue

            reply = message.content or ""
            print("AI:", reply)
            messages.append({"role": "assistant", "content": reply})
            messages = trim_conversation_history(messages, guardrail_config)
            break
        else:
            print("AI: I hit the tool-iteration limit for this request.")


if __name__ == "__main__":
    main()
