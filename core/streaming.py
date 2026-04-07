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
    tool_calls: list[Any] = []

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
            tool_calls.extend([tc for tc in delta.tool_calls if tc is not None])

    print()
    return collected_content, tool_calls

