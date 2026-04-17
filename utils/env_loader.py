"""Load environment variables from ~/.bashrc"""

import os
import re
import subprocess
from pathlib import Path


def load_env_from_bashrc():
    """Load environment variables from ~/.bashrc file."""
    bashrc_path = Path.home() / ".bashrc"

    if not bashrc_path.exists():
        return

    # Read bashrc file
    with open(bashrc_path, 'r') as f:
        content = f.read()

    # Find export statements with patterns like:
    # export OPENAI_API_KEY="xxxx"
    # export OPENAI_BASE_URL="xxxx"
    export_pattern = r'export\s+(\w+)=["\']?([^"\']*)["\']?'

    for match in re.finditer(export_pattern, content):
        var_name = match.group(1)
        var_value = match.group(2).strip()

        # Only set if not already set
        if var_name not in os.environ and var_value:
            os.environ[var_name] = var_value
            print(f"Loaded {var_name} from ~/.bashrc")


def get_openai_credentials():
    """Get OpenAI API key and base URL from environment."""
    load_env_from_bashrc()

    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("BEDROCK_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL", "https://bedrock-mantle.us-east-1.api.aws/v1")

    return api_key, base_url

