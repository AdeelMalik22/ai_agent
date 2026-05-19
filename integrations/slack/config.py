"""Slack configuration and constants."""

import os
from dataclasses import dataclass


@dataclass
class SlackConfig:
    """Slack integration configuration."""

    # Slack credentials (from environment)
    bot_token: str = os.getenv("SLACK_BOT_TOKEN", "")
    signing_secret: str = os.getenv("SLACK_SIGNING_SECRET", "")

    # Bedrock agent settings
    workspace_root: str = os.getenv("WORKSPACE_ROOT", os.getcwd())
    max_file_size_kb: int = int(os.getenv("MAX_FILE_SIZE", "100"))

    # Agent behavior
    max_message_length: int = 4000  # Slack message limit
    max_response_chunks: int = 10   # Max chunks before truncating
    stream_to_slack: bool = True    # Stream responses in real-time

    # Timeouts
    request_timeout: int = 3  # Slack API timeout in seconds
    agent_timeout: int = 60   # Agent response timeout in seconds

    def validate(self) -> tuple[bool, str]:
        """Validate configuration."""
        if not self.bot_token:
            return False, "SLACK_BOT_TOKEN not set in environment"
        if not self.signing_secret:
            return False, "SLACK_SIGNING_SECRET not set in environment"
        if not self.bot_token.startswith("xoxb-"):
            return False, "Invalid bot token format (should start with xoxb-)"
        return True, "Configuration valid"


def get_slack_config() -> SlackConfig:
    """Get Slack configuration from environment."""
    return SlackConfig()

