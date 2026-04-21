"""Streaming helpers for model output."""

from __future__ import annotations

import time
from typing import Any, cast

from openai import OpenAI

from tools import HANDOFF_TOOL, TOOLS


def stream_model_response(client: OpenAI, messages: list[dict], runtime_config: dict[str, Any]) -> tuple[str, list[Any]]:
    # noinspection PyArgumentList
    response = client.chat.completions.create(
        model=str(runtime_config["model"]),
        messages=cast(Any, messages),
        tools=cast(Any, TOOLS + [HANDOFF_TOOL]),
        tool_choice="auto",
        stream=True,
    )

    collected_content = ""
    # Track tool call information by index to handle streaming chunks
    # OpenAI sends tool calls across multiple chunks with index being consistent
    tool_data_by_index: dict[int, dict[str, Any]] = {}

    print("AI: ", end="", flush=True)
    for chunk in response:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta

        if delta.content:
            collected_content += delta.content
            print(delta.content, end="", flush=True)
            time.sleep(runtime_config["stream_delay"])

        if delta.tool_calls:
            for tc in delta.tool_calls:
                if tc is None:
                    continue

                # Get the index - this is consistent across chunks for the same tool call
                index = getattr(tc, "index", None)
                if index is None:
                    continue

                # Initialize tool data for this index if not exists
                if index not in tool_data_by_index:
                    tool_data_by_index[index] = {
                        "id": getattr(tc, "id", None),
                        "type": getattr(tc, "type", "function"),
                        "function": {
                            "name": "",
                            "arguments": "",
                        }
                    }

                # Update ID if this chunk has one (first chunk usually does)
                if getattr(tc, "id", None) is not None:
                    tool_data_by_index[index]["id"] = tc.id

                # Merge in function details
                if hasattr(tc, "function") and tc.function is not None:
                    if hasattr(tc.function, "name") and tc.function.name:
                        tool_data_by_index[index]["function"]["name"] = tc.function.name
                    if hasattr(tc.function, "arguments") and tc.function.arguments:
                        tool_data_by_index[index]["function"]["arguments"] += tc.function.arguments

    print()

    # Reconstruct tool calls from collected data
    tool_calls: list[Any] = []
    for index in sorted(tool_data_by_index.keys()):
        data = tool_data_by_index[index]

        # Only include if we have a function name and ID
        if not data["id"] or not data["function"]["name"]:
            continue

        # Create a simple object that has the attributes we need
        class ToolCall:
            def __init__(self, tool_id, func_name, arguments):
                self.id = tool_id
                self.type = "function"
                self.function = type('Function', (), {
                    'name': func_name,
                    'arguments': arguments
                })()

            def model_dump(self):
                return {
                    "id": self.id,
                    "type": self.type,
                    "function": {
                        "name": self.function.name,
                        "arguments": self.function.arguments,
                    }
                }

        tc = ToolCall(data["id"], data["function"]["name"], data["function"]["arguments"])
        tool_calls.append(tc)

    return collected_content, tool_calls

