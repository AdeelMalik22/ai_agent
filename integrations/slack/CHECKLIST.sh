#!/usr/bin/env bash
# Slack Integration Quick Setup Checklist
# Copy this to setup_slack.sh and follow the steps

echo "=================================="
echo "Slack Integration Setup Checklist"
echo "=================================="
echo ""

# Color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}STEP 1: Install Dependencies${NC}"
echo "[ ] Run: pip install -r integrations/slack/requirements.txt"
echo ""

echo -e "${BLUE}STEP 2: Create Slack App${NC}"
echo "[ ] Go to https://api.slack.com/apps"
echo "[ ] Click 'Create New App'"
echo "[ ] Select 'From scratch'"
echo "[ ] Name: 'Bedrock AI Agent'"
echo "[ ] Select your workspace"
echo "[ ] Click 'Create App'"
echo ""

echo -e "${BLUE}STEP 3: Configure Slash Command${NC}"
echo "[ ] In left sidebar: 'Slash Commands'"
echo "[ ] Click 'Create New Command'"
echo "[ ] Command: /agent"
echo "[ ] Request URL: https://your-domain/slack/slash"
echo "[ ] Short Description: Ask Bedrock AI Agent a question"
echo "[ ] Click 'Save'"
echo ""

echo -e "${BLUE}STEP 4: Configure Events${NC}"
echo "[ ] In left sidebar: 'Event Subscriptions'"
echo "[ ] Toggle 'Enable Events' ON"
echo "[ ] Request URL: https://your-domain/slack/events"
echo "[ ] Subscribe to: app_mention"
echo "[ ] Click 'Save Changes'"
echo ""

echo -e "${BLUE}STEP 5: Add Bot Scopes${NC}"
echo "[ ] In left sidebar: 'OAuth & Permissions'"
echo "[ ] Add scopes:"
echo "    - chat:write"
echo "    - commands"
echo "    - app_mentions:read"
echo "[ ] Click 'Save'"
echo ""

echo -e "${BLUE}STEP 6: Install App${NC}"
echo "[ ] Click 'Install App' in left sidebar"
echo "[ ] Click 'Install to Workspace'"
echo "[ ] Review permissions"
echo "[ ] Click 'Allow'"
echo "[ ] COPY: Bot User OAuth Token (starts with xoxb-)"
echo ""

echo -e "${BLUE}STEP 7: Get Signing Secret${NC}"
echo "[ ] Go to 'Basic Information'"
echo "[ ] Scroll to 'App Credentials'"
echo "[ ] COPY: Signing Secret"
echo ""

echo -e "${BLUE}STEP 8: Set Environment Variables${NC}"
echo "[ ] export SLACK_BOT_TOKEN='xoxb-...'"
echo "[ ] export SLACK_SIGNING_SECRET='...'"
echo "[ ] export BEDROCK_API_KEY='...'"
echo "[ ] export OPENAI_BASE_URL='https://bedrock-mantle.us-east-1.api.aws/v1'"
echo "[ ] export MODEL_ID='anthropic.claude-3-sonnet-20240229-v1:0'"
echo "[ ] export WORKSPACE_ROOT='${HOME}/bed_rock'"
echo ""

echo -e "${BLUE}STEP 9: Install Slack SDK${NC}"
echo "[ ] pip install -r integrations/slack/requirements.txt"
echo ""

echo -e "${BLUE}STEP 10: Start Server${NC}"
echo "[ ] python integrations/slack/run.py"
echo "[ ] You should see: 'Uvicorn running on http://0.0.0.0:8080'"
echo ""

echo -e "${BLUE}STEP 11: Expose Locally (Development Only)${NC}"
echo "[ ] Download ngrok: https://ngrok.com/download"
echo "[ ] In new terminal: ngrok http 8080"
echo "[ ] You'll get: https://abc123.ngrok.io"
echo "[ ] Copy this URL"
echo ""

echo -e "${BLUE}STEP 12: Update Slack URLs${NC}"
echo "[ ] Go back to Slack API Dashboard"
echo "[ ] Update Slash Command URL to: https://abc123.ngrok.io/slack/slash"
echo "[ ] Update Events URL to: https://abc123.ngrok.io/slack/events"
echo "[ ] Click 'Save'"
echo ""

echo -e "${BLUE}STEP 13: Test in Slack${NC}"
echo "[ ] Open your Slack workspace"
echo "[ ] Try: /agent What is Python?"
echo "[ ] Try: @Bedrock AI Agent Hello!"
echo "[ ] Check responses in Slack"
echo ""

echo -e "${GREEN}✅ Setup Complete!${NC}"
echo ""
echo "Next Steps:"
echo "1. Read: integrations/slack/SETUP.md (for full details)"
echo "2. Deploy: Use Docker or cloud platform (see SETUP.md)"
echo "3. Monitor: Check logs and status endpoint"
echo ""
echo "Useful URLs:"
echo "- Health Check: http://localhost:8080/health"
echo "- Status: http://localhost:8080/admin/status"
echo "- API Docs: http://localhost:8080/docs"
echo ""

