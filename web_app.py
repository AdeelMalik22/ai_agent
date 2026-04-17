"""Streamlit web UI for the AI Agent System"""
import os
import sys
import subprocess
from pathlib import Path
import streamlit as st
import re
sys.path.insert(0, str(Path(__file__).parent))
from openai import OpenAI
from config.settings import load_config
from core.session import initialize_session
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
    extract_tool_names,
    is_valid_response,
    create_tool_error_message,
)
from tools import HANDOFF_TOOL, TOOLS
def load_env_from_bashrc():
    """Load environment variables from ~/.bashrc file using bash."""
    bashrc_path = Path.home() / ".bashrc"

    if not bashrc_path.exists():
        return False

    try:
        # Use bash to source bashrc and export all variables
        result = subprocess.run(
            f'source ~/.bashrc && env | grep -E "OPENAI_|BEDROCK_"',
            shell=True,
            capture_output=True,
            text=True,
            executable="/bin/bash"
        )

        if result.returncode != 0:
            return False

        found = False
        for line in result.stdout.strip().split('\n'):
            if '=' in line:
                var_name, var_value = line.split('=', 1)
                if var_name and var_value:
                    os.environ[var_name] = var_value
                    found = True

        return found
    except Exception as e:
        st.error(f"Error loading environment: {str(e)}")
        return False
def build_client() -> OpenAI:
    """Build OpenAI/Bedrock client from environment."""
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("BEDROCK_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL", "https://bedrock-mantle.us-east-1.api.aws/v1")

    if not api_key:
        st.error("❌ API Key not found! Please check ~/.bashrc")
        st.stop()

    return OpenAI(
        api_key=api_key,
        base_url=base_url,
    )
def stream_model_response_streamlit(client: OpenAI, messages: list, runtime_config: dict, message_placeholder):
    """Stream model response directly to Streamlit placeholder."""
    response = client.chat.completions.create(
        model=str(runtime_config["model"]),
        messages=messages,
        tools=TOOLS + [HANDOFF_TOOL],
        tool_choice="auto",
        stream=True,
    )
    collected_content = ""
    tool_calls = []
    for chunk in response:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        if delta.content:
            collected_content += delta.content
            message_placeholder.markdown(collected_content)
        if delta.tool_calls:
            tool_calls.extend([tc for tc in delta.tool_calls if tc is not None])
    return collected_content, tool_calls
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
    for tool_call in tool_calls:
        # Skip invalid tool calls
        if tool_call is None or getattr(tool_call, 'id', None) is None:
            continue

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
    """Process complete model response (content + tool calls)."""
    if tool_calls:
        # Filter tool_calls to only include those with valid IDs
        valid_tool_calls = [
            tc.model_dump() for tc in tool_calls
            if tc is not None and getattr(tc, 'id', None) is not None
        ]

        if valid_tool_calls:
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
            # No valid tool calls, treat as regular response
            reply = guard_assistant_output(collected_content, output_guardrail_config)
            messages.append({"role": "assistant", "content": reply})
            messages = trim_conversation_history(messages, guardrail_config)
            return active_agent, handoffs_this_turn, False
    else:
        reply = guard_assistant_output(collected_content, output_guardrail_config)
        messages.append({"role": "assistant", "content": reply})
        messages = trim_conversation_history(messages, guardrail_config)
        return active_agent, handoffs_this_turn, False
def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="AI Agent Assistant",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    load_env_from_bashrc()
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("BEDROCK_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    if not api_key:
        st.error("❌ OpenAI/Bedrock API key not found in ~/.bashrc")
        st.info("Please set OPENAI_API_KEY or BEDROCK_API_KEY in your ~/.bashrc file")
        return
    st.title("🤖 AI Agent Assistant")
    st.markdown("Ask questions and get intelligent responses with multi-agent support")
    with st.sidebar:
        st.header("Configuration")
        if api_key:
            st.success("✅ API Key Loaded")
        if base_url:
            st.info(f"📍 Base URL: {base_url[:50]}...")
        st.divider()
        st.header("Settings")
        debug_mode = st.checkbox("Debug Mode", value=False)
        max_iterations = st.slider("Max Iterations", 1, 20, 8)
        stream_delay = st.slider("Stream Delay (seconds)", 0.0, 0.5, 0.05)
        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.session_state.known_tools = set()
            st.session_state.active_agent = "general"
            st.rerun()
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.known_tools = set()
        st.session_state.active_agent = "general"
        st.session_state.guardrail_config = None
        st.session_state.output_guardrail_config = None
        st.session_state.runtime_config = None
        st.session_state.client = None
        st.session_state.guardrail_config, st.session_state.output_guardrail_config, st.session_state.runtime_config = load_config()
        st.session_state.runtime_config["debug_handoffs"] = debug_mode
        st.session_state.runtime_config["max_iterations"] = max_iterations
        st.session_state.runtime_config["stream_delay"] = stream_delay
        messages, known_tools, active_agent = initialize_session()
        st.session_state.messages = messages
        st.session_state.known_tools = known_tools
        st.session_state.active_agent = active_agent
        st.session_state.client = build_client()
    st.session_state.runtime_config["debug_handoffs"] = debug_mode
    st.session_state.runtime_config["max_iterations"] = max_iterations
    st.session_state.runtime_config["stream_delay"] = stream_delay
    st.divider()
    for message in st.session_state.messages:
        if message["role"] == "system":
            continue
        if message["role"] == "user":
            with st.chat_message("user", avatar="👤"):
                st.markdown(message["content"])
        elif message["role"] == "assistant":
            with st.chat_message("assistant", avatar="🤖"):
                content = message.get("content", "")
                st.markdown(content)
                if "tool_calls" in message:
                    with st.expander("📋 Tool Calls", expanded=False):
                        for tool_call in message["tool_calls"]:
                            st.code(str(tool_call), language="json")
        elif message["role"] == "tool":
            with st.chat_message("tool", avatar="🔧"):
                content = message.get("content", "")
                if content.strip().startswith(("{", "[")):
                    st.code(content, language="json")
                else:
                    st.code(content, language="plaintext")
    st.divider()
    col1, col2 = st.columns([0.9, 0.1])
    with col1:
        user_input = st.text_input(
            "Ask a question...",
            placeholder="Type your question here",
            key="user_input_field"
        )
    with col2:
        send_button = st.button("Send", use_container_width=True)
    if send_button and user_input:
        user_input = user_input.strip()
        if user_input.lower() in {"exit", "quit", "bye"}:
            st.info("👋 Goodbye!")
            return
        user_check = validate_user_input(user_input, st.session_state.guardrail_config)
        if not user_check.allowed:
            st.error(f"❌ Input rejected: {user_check.reason}")
            return
        repeat_reason = validate_recent_user_repetition(
            st.session_state.messages,
            user_check.normalized_text,
            st.session_state.guardrail_config
        )
        if repeat_reason:
            st.warning(f"⚠️ {repeat_reason}")
            return
        st.session_state.messages.append({"role": "user", "content": user_check.normalized_text})
        st.session_state.messages = trim_conversation_history(
            st.session_state.messages,
            st.session_state.guardrail_config
        )
        st.rerun()
    if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
        with st.chat_message("assistant", avatar="🤖"):
            message_placeholder = st.empty()
            handoffs_this_turn = 0
            for iteration in range(st.session_state.runtime_config["max_iterations"]):
                try:
                    collected_content, tool_calls = stream_model_response_streamlit(
                        st.session_state.client,
                        st.session_state.messages,
                        st.session_state.runtime_config,
                        message_placeholder
                    )
                    if not is_valid_response(collected_content, tool_calls):
                        st.warning("[WARNING] Empty response from model")
                        continue
                    st.session_state.active_agent, handoffs_this_turn, should_continue = process_model_response(
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
                    if not should_continue:
                        break
                except Exception as e:
                    st.error(f"[ERROR] Iteration {iteration} failed: {str(e)}")
                    if iteration >= st.session_state.runtime_config["max_iterations"] - 1:
                        st.warning("⚠️ I hit the tool-iteration limit for this request.")
                    break
            else:
                st.warning("⚠️ I hit the tool-iteration limit for this request.")
if __name__ == "__main__":
    main()
