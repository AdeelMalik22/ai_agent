#!/usr/bin/env python
"""Quick start script for Slack integration.

This script starts the Slack integration server.
Make sure you have set the required environment variables first!

Environment variables needed:
    SLACK_BOT_TOKEN: xoxb-...
    SLACK_SIGNING_SECRET: ...
    BEDROCK_API_KEY: ...
    OPENAI_BASE_URL: ...
    MODEL_ID: ...
"""

import os
import sys
import logging

# Check environment variables
required_vars = [
    "SLACK_BOT_TOKEN",
    "SLACK_SIGNING_SECRET",
    "BEDROCK_API_KEY",
    "OPENAI_BASE_URL",
    "MODEL_ID"
]

missing_vars = [var for var in required_vars if not os.getenv(var)]

if missing_vars:
    print("❌ Missing required environment variables:")
    for var in missing_vars:
        print(f"   - {var}")
    print("\nSet them using:")
    print("   export SLACK_BOT_TOKEN='xoxb-...'")
    print("   export SLACK_SIGNING_SECRET='...'")
    print("   # etc.")
    sys.exit(1)

print("✅ All required environment variables are set!")
print()

# Print configuration summary
print("=" * 60)
print("Slack Integration Configuration")
print("=" * 60)
print(f"Bot Token: {os.getenv('SLACK_BOT_TOKEN')[:20]}...")
print(f"Model: {os.getenv('MODEL_ID')}")
print(f"Workspace Root: {os.getenv('WORKSPACE_ROOT', 'default')}")
print()

# Start the app
if __name__ == "__main__":
    import uvicorn
    from integrations.slack.slack_app import app

    port = int(os.getenv("SLACK_INTEGRATION_PORT", "8080"))

    print(f"🚀 Starting Slack integration server on port {port}...")
    print()
    print("Available endpoints:")
    print(f"  - Health:    http://localhost:{port}/health")
    print(f"  - Status:    http://localhost:{port}/admin/status")
    print(f"  - Events:    http://localhost:{port}/slack/events")
    print(f"  - Slash:     http://localhost:{port}/slack/slash")
    print(f"  - Docs:      http://localhost:{port}/docs")
    print()
    print("To expose locally (for Slack webhooks):")
    print("  ngrok http 8080")
    print()

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )

