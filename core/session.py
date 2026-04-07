"""Session initialization helpers."""

from __future__ import annotations

import os

from system_prompt import AGENT_PROMPTS
from tools import HANDOFF_TOOL, TOOLS


def initialize_session() -> tuple[list[dict], set[str], str]:
    active_agent = os.getenv("DEFAULT_AGENT", "general")
    if active_agent not in AGENT_PROMPTS:
        active_agent = "general"

    messages: list[dict] = [{"role": "system", "content": AGENT_PROMPTS[active_agent]}]
    known_tools = {tool["function"]["name"] for tool in TOOLS}
    known_tools.add(HANDOFF_TOOL["function"]["name"])

    return messages, known_tools, active_agent

