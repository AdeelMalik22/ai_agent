"""Slack integration handler for Bedrock AI Agent."""

import json
import logging
import os
import sys
from typing import Optional, Any

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from openai import OpenAI

from config.settings import load_config
from core.session import initialize_session
from core.streaming import stream_model_response
from guardrails.input_guardrils import validate_user_input, trim_conversation_history
from guardrails.output_guardrils import guard_assistant_output
from system_prompt import AGENT_PROMPTS
from utils.tooling import (
    validate_single_tool_call,
    execute_single_tool,
    is_valid_response,
    create_tool_error_message,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class SlackHandler:
    """Handle Slack integration with Bedrock AI Agent."""

    def __init__(self, bot_token: str, signing_secret: str):
        """Initialize Slack handler.

        Args:
            bot_token: Slack bot token (xoxb-...)
            signing_secret: Slack signing secret for request verification
        """
        self.client = WebClient(token=bot_token)
        self.signing_secret = signing_secret

        # Initialize agent session
        self.guardrail_config, self.output_guardrail_config, self.runtime_config = load_config()
        self.messages, self.known_tools, self.active_agent = initialize_session()

        # Initialize OpenAI/Bedrock client
        self.ai_client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY") or os.getenv("BEDROCK_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL", "https://bedrock-mantle.us-east-1.api.aws/v1"),
        )

        logger.info("SlackHandler initialized successfully")

    def verify_signature(self, timestamp: str, signature: str, body: str) -> bool:
        """Verify Slack request signature.

        Args:
            timestamp: Request timestamp from Slack
            signature: Signature from Slack header
            body: Raw request body

        Returns:
            True if signature is valid
        """
        import hmac
        import hashlib

        if abs(int(timestamp) - __import__('time').time()) > 300:
            logger.warning("Request timestamp too old (potential replay attack)")
            return False

        basestring = f"v0:{timestamp}:{body}"
        my_signature = "v0=" + hmac.new(
            self.signing_secret.encode(),
            basestring.encode(),
            hashlib.sha256
        ).hexdigest()

        return hmac.compare_digest(my_signature, signature)

    def handle_slash_command(self, command: str, text: str, user_id: str, channel_id: str,
                            response_url: str, trigger_id: str) -> dict:
        """Handle Slack slash command.

        Args:
            command: Slash command (e.g., /agent)
            text: Command text/arguments
            user_id: User ID who triggered command
            channel_id: Channel where command was issued
            response_url: URL to post response to
            trigger_id: Trigger ID for modals

        Returns:
            Response dict for Slack
        """
        logger.info(f"Slash command received: {command} from {user_id} with text: {text}")

        if not text.strip():
            return {
                "response_type": "ephemeral",
                "text": "Usage: `/agent <your question or command>`\n\nExamples:\n• `/agent What is Python?`\n• `/agent Review my code`\n• `/agent Handoff to coder`"
            }

        # Validate input
        user_check = validate_user_input(text, self.guardrail_config)
        if not user_check.allowed:
            return {
                "response_type": "ephemeral",
                "text": f"❌ Input rejected: {user_check.reason}"
            }

        # Immediate acknowledgment
        ack_response = {
            "response_type": "in_channel",
            "text": f"🤖 Processing your request...\n> {user_check.normalized_text}"
        }

        try:
            # Execute agent in background (non-blocking)
            self._process_agent_request(
                user_check.normalized_text,
                channel_id,
                user_id,
                response_url
            )
        except Exception as e:
            logger.error(f"Error processing agent request: {e}")
            return {
                "response_type": "ephemeral",
                "text": f"❌ Error: {str(e)}"
            }

        return ack_response

    def _process_agent_request(self, user_input: str, channel_id: str, user_id: str,
                              response_url: str) -> None:
        """Process user request through agent.

        Args:
            user_input: User's question/request
            channel_id: Slack channel ID
            user_id: Slack user ID
            response_url: URL to post response
        """
        try:
            # Add to conversation history
            self.messages.append({"role": "user", "content": user_input})
            self.messages = trim_conversation_history(self.messages, self.guardrail_config)

            # Get agent response
            collected_content, tool_calls = stream_model_response(
                self.ai_client,
                self.messages,
                self.runtime_config
            )

            if not is_valid_response(collected_content, tool_calls):
                response_text = "⚠️ Empty response from model. Please try again."
                self._send_response(response_url, response_text)
                return

            # Process tool calls if any
            if tool_calls:
                self._process_tool_calls(tool_calls, response_url)

            # Guard and send final response
            final_response = guard_assistant_output(collected_content, self.output_guardrail_config)
            self.messages.append({"role": "assistant", "content": final_response})
            self.messages = trim_conversation_history(self.messages, self.guardrail_config)

            # Send to Slack
            self._send_response(response_url, final_response)

        except Exception as e:
            logger.error(f"Error in agent request processing: {e}")
            error_msg = f"❌ Error: {str(e)[:100]}"
            self._send_response(response_url, error_msg)

    def _process_tool_calls(self, tool_calls: list, response_url: str) -> None:
        """Process tool calls from agent.

        Args:
            tool_calls: List of tool calls from model
            response_url: Slack response URL
        """
        valid_tool_calls = [
            tc for tc in tool_calls
            if tc is not None and getattr(tc, "id", None) is not None
        ]

        for tool_call in valid_tool_calls:
            is_valid, error_msg, parsed_args = validate_single_tool_call(
                tool_call, self.known_tools, self.guardrail_config
            )

            if not is_valid:
                error_message = create_tool_error_message(
                    tool_call, error_msg, self.output_guardrail_config
                )
                self.messages.append(error_message)
                continue

            # Execute tool
            result, new_agent, _ = execute_single_tool(
                tool_call,
                parsed_args,
                self.active_agent,
                0,
                self.runtime_config["max_handoffs_per_turn"],
                self.output_guardrail_config,
                allow_system_read=self.runtime_config.get("allow_system_file_read", False),
            )

            if new_agent != self.active_agent:
                self.active_agent = new_agent
                self.messages[0] = {"role": "system", "content": AGENT_PROMPTS[self.active_agent]}

            self.messages.append({
                "role": "tool",
                "tool_call_id": getattr(tool_call, "id", "unknown"),
                "content": result,
            })

            # Notify user about tool execution
            tool_name = getattr(tool_call.function, "name", "unknown")
            tool_msg = f"🔧 Used tool: `{tool_name}`"
            self._send_response(response_url, tool_msg)

    def _send_response(self, response_url: str, text: str) -> None:
        """Send response to Slack via response URL.

        Args:
            response_url: Slack response URL
            text: Response text (supports Slack markdown)
        """
        import requests

        try:
            # Truncate if too long
            if len(text) > 4000:
                text = text[:3990] + "\n... (truncated)"

            payload = {
                "response_type": "in_channel",
                "text": text
            }

            response = requests.post(response_url, json=payload, timeout=5)
            response.raise_for_status()
            logger.info("Response sent to Slack successfully")

        except Exception as e:
            logger.error(f"Error sending response to Slack: {e}")

    def handle_app_mention(self, event: dict) -> Optional[str]:
        """Handle when bot is mentioned in channel.

        Args:
            event: Slack event payload

        Returns:
            Response text or None
        """
        text = event.get("text", "").replace(f"<@{event.get('user')}>", "").strip()
        channel_id = event.get("channel")
        user_id = event.get("user")
        thread_ts = event.get("thread_ts", event.get("ts"))

        logger.info(f"App mention received from {user_id}: {text}")

        if not text:
            return "Hi! Ask me anything or try `/agent help`"

        # Validate input
        user_check = validate_user_input(text, self.guardrail_config)
        if not user_check.allowed:
            return f"❌ Input rejected: {user_check.reason}"

        try:
            # Add to conversation
            self.messages.append({"role": "user", "content": user_check.normalized_text})
            self.messages = trim_conversation_history(self.messages, self.guardrail_config)

            # Get response
            collected_content, tool_calls = stream_model_response(
                self.ai_client,
                self.messages,
                self.runtime_config
            )

            if not is_valid_response(collected_content, tool_calls):
                return "⚠️ Empty response from model"

            # Process tools if needed
            if tool_calls:
                self._process_tool_calls_for_mention(tool_calls)

            # Guard and return response
            final_response = guard_assistant_output(collected_content, self.output_guardrail_config)
            self.messages.append({"role": "assistant", "content": final_response})
            self.messages = trim_conversation_history(self.messages, self.guardrail_config)

            return final_response

        except Exception as e:
            logger.error(f"Error handling app mention: {e}")
            return f"❌ Error: {str(e)[:100]}"

    def _process_tool_calls_for_mention(self, tool_calls: list) -> None:
        """Process tool calls for app mentions (similar to slash commands)."""
        valid_tool_calls = [
            tc for tc in tool_calls
            if tc is not None and getattr(tc, "id", None) is not None
        ]

        for tool_call in valid_tool_calls:
            is_valid, error_msg, parsed_args = validate_single_tool_call(
                tool_call, self.known_tools, self.guardrail_config
            )

            if not is_valid:
                error_message = create_tool_error_message(
                    tool_call, error_msg, self.output_guardrail_config
                )
                self.messages.append(error_message)
                continue

            result, new_agent, _ = execute_single_tool(
                tool_call,
                parsed_args,
                self.active_agent,
                0,
                self.runtime_config["max_handoffs_per_turn"],
                self.output_guardrail_config,
                allow_system_read=self.runtime_config.get("allow_system_file_read", False),
            )

            if new_agent != self.active_agent:
                self.active_agent = new_agent
                self.messages[0] = {"role": "system", "content": AGENT_PROMPTS[self.active_agent]}

            self.messages.append({
                "role": "tool",
                "tool_call_id": getattr(tool_call, "id", "unknown"),
                "content": result,
            })

    def reset_conversation(self) -> None:
        """Reset conversation history."""
        self.messages, self.known_tools, self.active_agent = initialize_session()
        logger.info("Conversation history reset")

