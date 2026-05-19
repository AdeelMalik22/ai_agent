"""FastAPI application for Slack integration with Bedrock AI Agent."""

import logging
import os
from typing import Optional

from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel, Field

from .slack_handler import SlackHandler
from .config import get_slack_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Bedrock AI Agent - Slack Integration",
    description="Slack bot for Bedrock OpenAI-Compatible Tool Agent",
    version="1.0.0"
)

# Initialize Slack handler (lazy loaded)
_slack_handler: Optional[SlackHandler] = None


def get_handler() -> SlackHandler:
    """Get or initialize Slack handler."""
    global _slack_handler
    if _slack_handler is None:
        config = get_slack_config()
        is_valid, error_msg = config.validate()
        if not is_valid:
            raise RuntimeError(f"Invalid Slack configuration: {error_msg}")
        _slack_handler = SlackHandler(config.bot_token, config.signing_secret)
    return _slack_handler


# ============================================================================
# Request Models
# ============================================================================

class SlashCommandPayload(BaseModel):
    """Slack slash command payload."""
    token: str
    team_id: str
    team_domain: str
    channel_id: str
    channel_name: str
    user_id: str
    user_name: str
    command: str
    text: str
    api_app_id: str
    response_url: str
    trigger_id: str

    class Config:
        extra = "allow"


class EventPayload(BaseModel):
    """Slack event payload."""
    token: str
    team_id: str
    api_app_id: str
    event: dict
    type: str
    event_id: str
    event_time: int

    class Config:
        extra = "allow"


class UrlVerification(BaseModel):
    """Slack URL verification payload."""
    type: str
    challenge: str
    token: str


# ============================================================================
# Health Check
# ============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "Bedrock AI Agent - Slack Integration",
        "version": "1.0.0"
    }


# ============================================================================
# Slack Endpoints
# ============================================================================

@app.post("/slack/events")
async def handle_slack_event(request: Request):
    """Handle Slack events (URL verification, mentions, etc).

    Slack sends different event types:
    - url_verification: For endpoint verification
    - app_mention: When bot is mentioned
    - message: Regular messages
    """
    try:
        # Verify request signature
        timestamp = request.headers.get("X-Slack-Request-Timestamp", "")
        signature = request.headers.get("X-Slack-Signature", "")
        body = await request.body()
        body_str = body.decode('utf-8')

        handler = get_handler()

        if not handler.verify_signature(timestamp, signature, body_str):
            logger.warning("Invalid Slack request signature")
            raise HTTPException(status_code=401, detail="Invalid signature")

        # Parse payload
        payload = await request.json()

        # Handle URL verification
        if payload.get("type") == "url_verification":
            logger.info("URL verification requested")
            return {"challenge": payload.get("challenge")}

        # Handle events
        if payload.get("type") == "event_callback":
            event = payload.get("event", {})
            event_type = event.get("type")

            logger.info(f"Event received: {event_type}")

            if event_type == "app_mention":
                response_text = handler.handle_app_mention(event)
                if response_text:
                    try:
                        # Post response to thread
                        handler.client.chat_postMessage(
                            channel=event.get("channel"),
                            text=response_text,
                            thread_ts=event.get("thread_ts", event.get("ts"))
                        )
                    except Exception as e:
                        logger.error(f"Error posting message to Slack: {e}")

            return {"ok": True}

        return {"ok": True}

    except Exception as e:
        logger.error(f"Error handling Slack event: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/slack/slash")
async def handle_slash_command(payload: SlashCommandPayload):
    """Handle Slack slash commands.

    Slack calls this endpoint when user types a slash command like /agent
    """
    try:
        handler = get_handler()

        response = handler.handle_slash_command(
            command=payload.command,
            text=payload.text,
            user_id=payload.user_id,
            channel_id=payload.channel_id,
            response_url=payload.response_url,
            trigger_id=payload.trigger_id
        )

        return response

    except Exception as e:
        logger.error(f"Error handling slash command: {e}")
        return {
            "response_type": "ephemeral",
            "text": f"❌ Error processing command: {str(e)}"
        }


# ============================================================================
# Admin Endpoints
# ============================================================================

@app.post("/admin/reset-conversation")
async def reset_conversation():
    """Reset conversation history (admin endpoint).

    Use this to clear conversation context if needed.
    """
    try:
        handler = get_handler()
        handler.reset_conversation()
        return {"status": "success", "message": "Conversation history reset"}
    except Exception as e:
        logger.error(f"Error resetting conversation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/admin/status")
async def get_status():
    """Get integration status."""
    try:
        handler = get_handler()
        return {
            "status": "connected",
            "active_agent": handler.active_agent,
            "messages_in_history": len(handler.messages),
            "known_tools": len(handler.known_tools)
        }
    except Exception as e:
        logger.error(f"Error getting status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Root endpoint
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "Bedrock AI Agent - Slack Integration",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "slack_events": "/slack/events",
            "slack_slash": "/slack/slash",
            "admin_status": "/admin/status",
            "admin_reset": "/admin/reset-conversation",
            "docs": "/docs",
            "openapi": "/openapi.json"
        }
    }


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("SLACK_INTEGRATION_PORT", "8080"))
    uvicorn.run(app, host="0.0.0.0", port=port)

