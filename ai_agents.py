import json
import os
from openai import OpenAI

from system_prompt import AGENT_PROMPTS
from tools import TOOLS, run_tool, HANDOFF_TOOL, execute_handoff


def build_client() -> OpenAI:
    return OpenAI(
        api_key=os.getenv("OPENAI_API_KEY") or os.getenv("BEDROCK_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL", "https://bedrock-mantle.us-east-1.api.aws/v1"),
    )


def main() -> None:
    client = build_client()
    model = os.getenv("MODEL_ID", "openai.gpt-oss-120b")
    max_iterations = int(os.getenv("MAX_TOOL_ITERATIONS", "8"))
    max_handoffs_per_turn = int(os.getenv("MAX_HANDOFFS_PER_TURN", "2"))
    debug_handoffs = os.getenv("DEBUG_HANDOFFS", "1") == "1"
    active_agent = os.getenv("DEFAULT_AGENT", "general")
    if active_agent not in AGENT_PROMPTS:
        active_agent = "general"
    messages = [{"role": "system", "content": AGENT_PROMPTS[active_agent]}]

    while True:
        user_input = input("ask Question.....: ").strip()
        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            print("Bye")
            break

        messages.append({"role": "user", "content": user_input})
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
                    if tool_call.function.name == "handoff_to_agent":
                        old_agent = active_agent
                        result, new_agent, handoffs_this_turn = execute_handoff(
                            raw_arguments=tool_call.function.arguments,
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
                            raw_arguments=tool_call.function.arguments,
                        )
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": result,
                        }
                    )
                continue

            reply = message.content or ""
            print("AI:", reply)
            messages.append({"role": "assistant", "content": reply})
            break
        else:
            print("AI: I hit the tool-iteration limit for this request.")


if __name__ == "__main__":
    main()
