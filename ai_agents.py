import json
import os
from openai import OpenAI
from system_prompt import SYSTEM_PROMPT
from tools import TOOLS, run_tool



def build_client() -> OpenAI:
    return OpenAI(
        api_key=os.getenv("OPENAI_API_KEY") or os.getenv("BEDROCK_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL", "https://bedrock-mantle.us-east-1.api.aws/v1"),
    )


def main() -> None:
    client = build_client()
    model = os.getenv("MODEL_ID", "openai.gpt-oss-120b")

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    while True:
        user_input = input("ask Question.....: ").strip()
        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            print("Bye")
            break

        messages.append({"role": "user", "content": user_input})

        for _ in range(8):
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
            )

            message = response.choices[0].message
            print(message.tool_calls or [])
            tool_calls = message.tool_calls or []

            if tool_calls:
                assistant_message = {
                    "role": "assistant",
                    "content": message.content or "",
                    "tool_calls": [tc.model_dump() for tc in tool_calls],
                }
                messages.append(assistant_message)

                for tool_call in tool_calls:
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
