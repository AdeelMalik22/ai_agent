"""FastAPI server for Agent IDE extension.

Wraps the existing AI agent system from ai_agent.py to serve HTTP requests.
Maintains all guardrails, tool processing, and streaming logic.
"""

import logging
from typing import Any, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global state
_state = {
    "messages": [],
    "active_agent": None,
    "known_tools": set(),
    "client": None,
    "guardrail_config": None,
    "output_guardrail_config": None,
    "runtime_config": None,
}


def build_client() -> OpenAI:
    """Build OpenAI/Bedrock client from environment."""
    import os
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
    """Process all tool calls from model response."""
    valid_tool_calls = [
        tc for tc in tool_calls
        if tc is not None and getattr(tc, "id", None) is not None
    ]

    if runtime_config.get("debug_handoffs"):
        tool_names = extract_tool_names(valid_tool_calls)
        logger.info(f"Tools requested: {tool_names}")

    for tool_call in valid_tool_calls:
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
            runtime_config.get("max_handoffs_per_turn", 5),
            output_guardrail_config,
            allow_system_read=runtime_config.get("allow_system_file_read", False),
        )

        if new_agent != active_agent:
            active_agent = new_agent
            messages[0] = {"role": "system", "content": AGENT_PROMPTS[active_agent]}
            if runtime_config.get("debug_handoffs"):
                logger.info(f"Handoff to: {active_agent}")

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
) -> tuple[str, int, bool, list[Any]]:
    """Process complete model response.

    Returns: (active_agent, handoffs_count, should_continue, tool_calls)
    """
    if tool_calls:
        valid_tool_calls = [
            tc for tc in tool_calls
            if tc is not None and getattr(tc, "id", None) is not None
        ]

        if not valid_tool_calls:
            handle_assistant_reply(collected_content, messages, guardrail_config, output_guardrail_config)
            return active_agent, handoffs_this_turn, False, []

        assistant_message_with_tools = {
            "role": "assistant",
            "content": guard_assistant_output(collected_content, output_guardrail_config),
            "tool_calls": [tc.model_dump() if hasattr(tc, 'model_dump') else tc for tc in valid_tool_calls],
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
        return active_agent, handoffs_this_turn, should_continue, valid_tool_calls
    else:
        handle_assistant_reply(collected_content, messages, guardrail_config, output_guardrail_config)
        return active_agent, handoffs_this_turn, False, []


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup application state."""
    try:
        logger.info("Initializing Agent IDE backend...")
        _state["client"] = build_client()

        guardrail_config, output_guardrail_config, runtime_config = load_config()
        _state["guardrail_config"] = guardrail_config
        _state["output_guardrail_config"] = output_guardrail_config
        _state["runtime_config"] = runtime_config

        messages, known_tools, active_agent = initialize_session()
        _state["messages"] = messages
        _state["known_tools"] = known_tools
        _state["active_agent"] = active_agent

        logger.info("✓ Initialization successful")
    except Exception as e:
        logger.error(f"✗ Initialization failed: {e}", exc_info=True)
        raise

    yield

    logger.info("Shutting down Agent IDE backend...")


app = FastAPI(
    title="Agent IDE Backend",
    description="AI-powered VS Code extension backend",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    """Request model for chat endpoint"""
    message: str = Field(..., min_length=1, max_length=10000, description="User message")
    context: Optional[dict[str, Any]] = Field(
        default=None,
        description="Optional context from VS Code"
    )


class ChatResponse(BaseModel):
    """Response model for chat endpoint"""
    type: str = Field(..., description="Response type: message, tool_calls, error")
    content: Optional[str] = Field(default=None, description="Response text")
    tool_calls: Optional[list[Any]] = Field(default=None, description="Tool calls if any")
    agent: Optional[str] = Field(default=None, description="Current active agent")


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    """Main chat endpoint - processes user messages like the terminal version."""
    try:
        # Validate state
        if not _state["client"]:
            logger.error("Client not initialized")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Server not properly initialized"
            )

        logger.info(f"📨 User message: {req.message[:60]}...")

        # Step 1: Validate user input using guardrails
        user_check = validate_user_input(req.message, _state["guardrail_config"])
        if not user_check.allowed:
            logger.warning(f"Input rejected: {user_check.reason}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Input rejected: {user_check.reason}"
            )

        # Step 2: Check for repetition
        repeat_reason = validate_recent_user_repetition(
            _state["messages"],
            user_check.normalized_text,
            _state["guardrail_config"]
        )
        if repeat_reason:
            logger.warning(f"Repetition detected: {repeat_reason}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Input rejected: {repeat_reason}"
            )

        # Step 3: Add to message history
        normalized_text = user_check.normalized_text
        _state["messages"].append({"role": "user", "content": normalized_text})
        _state["messages"] = trim_conversation_history(
            _state["messages"],
            _state["guardrail_config"]
        )

        logger.info("✓ Input validation passed")

        # Step 4: Get model response
        handoffs_this_turn = 0
        max_iterations = _state["runtime_config"].get("max_iterations", 5)

        for iteration in range(max_iterations):
            try:
                logger.info(f"  Iteration {iteration + 1}/{max_iterations}...")

                collected_content, tool_calls = stream_model_response(
                    _state["client"],
                    _state["messages"],
                    _state["runtime_config"]
                )

                if not is_valid_response(collected_content, tool_calls):
                    logger.warning("Empty response, retrying...")
                    continue

                # Process the response
                active_agent, handoffs_this_turn, should_continue, processed_tools = process_model_response(
                    collected_content,
                    tool_calls,
                    _state["messages"],
                    _state["active_agent"],
                    handoffs_this_turn,
                    _state["known_tools"],
                    _state["guardrail_config"],
                    _state["output_guardrail_config"],
                    _state["runtime_config"],
                )

                _state["active_agent"] = active_agent

                if not should_continue:
                    # No more tool calls, return final response
                    logger.info(f"✓ Response complete (agent: {active_agent})")
                    return ChatResponse(
                        type="message",
                        content=collected_content,
                        agent=active_agent
                    )

            except Exception as e:
                logger.error(f"Iteration {iteration} failed: {e}", exc_info=True)
                if iteration >= max_iterations - 1:
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail="Hit maximum iterations limit"
                    )
                continue

        logger.error("Hit max iterations")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Hit the tool-iteration limit for this request"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


@app.post("/reset")
async def reset_session():
    """Reset the conversation and active agent."""
    try:
        logger.info("Resetting session...")
        messages, known_tools, active_agent = initialize_session()
        _state["messages"] = messages
        _state["known_tools"] = known_tools
        _state["active_agent"] = active_agent
        logger.info("✓ Session reset")
        return {"status": "reset", "agent": active_agent}
    except Exception as e:
        logger.error(f"Reset failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Reset failed: {str(e)}"
        )


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "initialized": _state["client"] is not None,
        "agent": _state["active_agent"],
        "messages_count": len(_state["messages"])
    }


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Agent IDE Backend",
        "version": "1.0.0",
        "agent": _state["active_agent"],
        "endpoints": {
            "chat": "POST /chat - Send message to agent",
            "reset": "POST /reset - Reset conversation",
            "health": "GET /health - Health check",
            "docs": "GET /docs - API documentation"
        }
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api_server:app",
        host="localhost",
        port=3000,
        reload=True,
        log_level="info"
    )
