"""Streamlit frontend for the AI agent system with full agent support."""

import os
import streamlit as st
from openai import OpenAI
from config.settings import load_config
from core.session import initialize_session
from core.streaming import stream_model_response
from guardrails.input_guardrils import (
    trim_conversation_history,
    validate_recent_user_repetition,
    validate_user_input,
)
from guardrails.output_guardrils import guard_assistant_output
from system_prompt import AGENT_PROMPTS
from utils.tooling import (
    validate_single_tool_call,
    execute_single_tool,
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
    tool_calls,
    messages,
    active_agent,
    handoffs_this_turn,
    known_tools,
    guardrail_config,
    output_guardrail_config,
    runtime_config,
):
    """Process all tool calls from model response."""
    valid_tool_calls = [
        tc for tc in tool_calls
        if tc is not None and getattr(tc, "id", None) is not None
    ]

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
            runtime_config["max_handoffs_per_turn"],
            output_guardrail_config,
        )

        if new_agent != active_agent:
            active_agent = new_agent
            messages[0] = {"role": "system", "content": AGENT_PROMPTS[active_agent]}

        messages.append({
            "role": "tool",
            "tool_call_id": getattr(tool_call, "id", "unknown"),
            "content": result,
        })
        messages = trim_conversation_history(messages, guardrail_config)

    return active_agent, True, handoffs_this_turn


def process_model_response(
    collected_content,
    tool_calls,
    messages,
    active_agent,
    handoffs_this_turn,
    known_tools,
    guardrail_config,
    output_guardrail_config,
    runtime_config,
):
    """Process complete model response."""
    if tool_calls:
        valid_tool_calls = [
            tc.model_dump() for tc in tool_calls
            if tc is not None and getattr(tc, "id", None) is not None
        ]

        if not valid_tool_calls:
            reply = guard_assistant_output(collected_content, output_guardrail_config)
            messages.append({"role": "assistant", "content": reply})
            messages = trim_conversation_history(messages, guardrail_config)
            return active_agent, handoffs_this_turn, False

        assistant_message_with_tools = {
            "role": "assistant",
            "content": guard_assistant_output(collected_content, output_guardrail_config),
            "tool_calls": valid_tool_calls,
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
        reply = guard_assistant_output(collected_content, output_guardrail_config)
        messages.append({"role": "assistant", "content": reply})
        messages = trim_conversation_history(messages, guardrail_config)
        return active_agent, handoffs_this_turn, False


def initialize_streamlit_session():
    """Initialize Streamlit session state."""
    if "messages" not in st.session_state:
        config = load_config()
        messages, known_tools, active_agent = initialize_session()
        st.session_state.messages = messages
        st.session_state.known_tools = known_tools
        st.session_state.active_agent = active_agent
        st.session_state.guardrail_config = config[0]
        st.session_state.output_guardrail_config = config[1]
        st.session_state.runtime_config = config[2]
        st.session_state.client = build_client()


def main():
    """Main Streamlit app."""
    st.set_page_config(
        page_title="AI Agent System",
        page_icon="🤖",
        layout="wide",
    )

    st.title("🤖 AI Agent System")
    st.markdown("Chat with AI specialists for coding, planning, and reviews")

    # Initialize session
    initialize_streamlit_session()

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        st.write(f"**Active Agent:** {st.session_state.active_agent.upper()}")

        if st.button("🔄 Reset Conversation"):
            st.session_state.clear()
            st.rerun()

    # Display chat history (excluding system messages, tool messages, and internal assistant tool-call steps)
    st.divider()
    for message in st.session_state.messages:
        if message.get("role") == "system":
            continue

        # Skip tool messages - don't display them in UI
        if message.get("role") == "tool":
            continue

        if message.get("role") == "user":
            with st.chat_message("user", avatar="👤"):
                st.write(message.get("content", ""))

        elif message.get("role") == "assistant":
            # Assistant messages that include tool calls are internal reasoning steps;
            # keep them in state for the model, but don't render them as extra bubbles.
            if message.get("tool_calls"):
                continue

            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(
                    guard_assistant_output(
                        message.get("content", ""),
                        st.session_state.output_guardrail_config,
                    )
                )


    # Input area
    user_input = st.chat_input("Ask me anything...")

    if user_input:
        user_input_text = getattr(user_input, "text", user_input)
        if not isinstance(user_input_text, str):
            user_input_text = str(user_input_text)

        # Validate input
        user_check = validate_user_input(user_input_text, st.session_state.guardrail_config)
        if not user_check.allowed:
            st.error(f"Input rejected: {user_check.reason}")
            st.stop()

        repeat_reason = validate_recent_user_repetition(
            st.session_state.messages,
            user_check.normalized_text,
            st.session_state.guardrail_config,
        )
        if repeat_reason:
            st.warning(repeat_reason)
            st.stop()

        # Add user message
        st.session_state.messages.append({"role": "user", "content": user_check.normalized_text})
        st.session_state.messages = trim_conversation_history(
            st.session_state.messages,
            st.session_state.guardrail_config,
        )

        # Display user message
        with st.chat_message("user", avatar="👤"):
            st.write(user_check.normalized_text)

        # Show thinking indicator
        thinking_placeholder = st.empty()
        thinking_placeholder.info("🤔 AI is thinking...")

        # Process with AI agent loop
        handoffs_this_turn = 0
        iteration = 0
        max_iterations = st.session_state.runtime_config["max_iterations"]

        response_displayed = False

        while iteration < max_iterations:
            try:
                # Get model response with streaming
                collected_content, tool_calls = stream_model_response(
                    st.session_state.client,
                    st.session_state.messages,
                    st.session_state.runtime_config,
                )

                if not is_valid_response(collected_content, tool_calls):
                    thinking_placeholder.warning("⚠️ Empty response from model")
                    break

                # Check if AI is using web search
                if tool_calls and any(
                    tc is not None and getattr(tc, "id", None) is not None
                    and getattr(tc.function, "name", "") == "web_search"
                    for tc in tool_calls
                ):
                    thinking_placeholder.info("🔍 Searching the web for latest information...")

                # If there are no tool calls, display the response and stop
                if not tool_calls or not any(tc is not None and getattr(tc, "id", None) is not None for tc in tool_calls):
                    final_reply = guard_assistant_output(
                        collected_content,
                        st.session_state.output_guardrail_config,
                    )

                    # Clear thinking indicator
                    thinking_placeholder.empty()

                    # Display final response
                    with st.chat_message("assistant", avatar="🤖"):
                        st.markdown(final_reply)

                    # Save to messages
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": final_reply
                    })
                    st.session_state.messages = trim_conversation_history(
                        st.session_state.messages,
                        st.session_state.guardrail_config,
                    )
                    response_displayed = True
                    break

                # Process tool calls (handoffs, web search, etc)
                if not any(getattr(tc.function, "name", "") == "web_search" for tc in tool_calls if tc is not None):
                    thinking_placeholder.info("🔄 Processing...")

                active_agent, handoffs_this_turn, should_continue = process_model_response(
                    collected_content,
                    tool_calls,
                    st.session_state.messages,
                    st.session_state.active_agent,
                    handoffs_this_turn,
                    st.session_state.known_tools,
                    st.session_state.guardrail_config,
                    st.session_state.output_guardrail_config,
                    st.session_state.runtime_config,
                )

                st.session_state.active_agent = active_agent

                if not should_continue:
                    thinking_placeholder.empty()
                    break

                # Update thinking message to show current status
                thinking_placeholder.info(f"🤔 Ai is thinking")

                iteration += 1

            except Exception as e:
                thinking_placeholder.error(f"❌ Error: {str(e)}")
                break

        # Clear thinking indicator if still showing
        if response_displayed:
            thinking_placeholder.empty()

        st.rerun()


if __name__ == "__main__":
    main()

