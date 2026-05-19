# 🎯 Slack Integration Setup Guide

Complete guide to integrating Bedrock AI Agent with Slack.

---

## 📋 Prerequisites

- ✅ Slack workspace (free or paid)
- ✅ Bedrock AI Agent running locally or deployed
- ✅ Python 3.10+
- ✅ Public URL or tunnel to your integration endpoint

---

## 🚀 Quick Start (15 minutes)

### Step 1: Create Slack App

1. Go to [Slack API Dashboard](https://api.slack.com/apps)
2. Click **"Create New App"**
3. Choose **"From scratch"**
4. Name: `Bedrock AI Agent`
5. Select your workspace
6. Click **"Create App"**

### Step 2: Configure Slack App

#### Enable Slash Commands

1. In left sidebar, go to **"Slash Commands"**
2. Click **"Create New Command"**
3. Fill in:
   - **Command:** `/agent`
   - **Request URL:** `https://your-domain.com/slack/slash`
   - **Short Description:** `Ask Bedrock AI Agent a question`
   - **Usage hint:** `[your question]`
4. Click **"Save"**

#### Enable Events

1. In left sidebar, go to **"Event Subscriptions"**
2. Toggle **"Enable Events"** to ON
3. Set **Request URL:** `https://your-domain.com/slack/events`
   - Slack will send a verification request
   - Your app must return the challenge
4. Subscribe to bot events:
   - `app_mention` - When bot is mentioned
   - `message.im` - Direct messages (optional)
5. Click **"Save Changes"**

#### Set Bot Token Scopes

1. In left sidebar, go to **"OAuth & Permissions"**
2. Scroll to **"Scopes"** section
3. Under **"Bot Token Scopes"**, add:
   - `chat:write` - Send messages
   - `commands` - Handle slash commands
   - `app_mentions:read` - Listen to mentions
   - `im:read` - Read direct messages (optional)
   - `channels:read` - Read channel info
4. Click **"Save"**

### Step 3: Install App

1. In left sidebar, go to **"Install App"**
2. Click **"Install to Workspace"**
3. Review permissions
4. Click **"Allow"**
5. Copy the **Bot User OAuth Token** (starts with `xoxb-`)

### Step 4: Get Signing Secret

1. Go to **"Basic Information"**
2. Scroll to **"App Credentials"**
3. Copy **Signing Secret**

---

## 🔧 Local Setup

### Install Dependencies

```bash
cd /home/enigmatix/bed_rock
pip install -r integrations/slack/requirements.txt
```

### Set Environment Variables

```bash
# Slack credentials (from Step 3 & 4 above)
export SLACK_BOT_TOKEN='xoxb-your-token-here'
export SLACK_SIGNING_SECRET='your-signing-secret-here'

# Bedrock credentials (existing)
export BEDROCK_API_KEY='your-bedrock-key'
export OPENAI_BASE_URL='https://bedrock-mantle.us-east-1.api.aws/v1'
export MODEL_ID='anthropic.claude-3-sonnet-20240229-v1:0'
export WORKSPACE_ROOT='/home/username/bed_rock'

# Integration settings
export SLACK_INTEGRATION_PORT='8080'
```

### Start Integration Server

```bash
python -m uvicorn integrations.slack.slack_app:app --host 0.0.0.0 --port 8080 --reload
```

You should see:
```
INFO:     Uvicorn running on http://0.0.0.0:8080
```

### Expose Locally (Development)

Use **ngrok** or **localtunnel** to expose your local server:

```bash
# Using ngrok (install from https://ngrok.com)
ngrok http 8080

# You'll get a URL like: https://abc123.ngrok.io
```

### Update Slack URLs

Go back to Slack API Dashboard:

1. **Slash Commands** → Update Request URL to: `https://your-ngrok-url/slack/slash`
2. **Event Subscriptions** → Update Request URL to: `https://your-ngrok-url/slack/events`

---

## 🐳 Docker Deployment

### Build Docker Image

```bash
cd /home/enigmatix/bed_rock

# Create Dockerfile for Slack integration
cat > Dockerfile.slack <<EOF
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
COPY integrations/slack/requirements.txt integrations/slack/requirements.txt

RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -r integrations/slack/requirements.txt

COPY . .

ENV PYTHONPATH=/app
ENV SLACK_INTEGRATION_PORT=8080

EXPOSE 8080

CMD ["python", "-m", "uvicorn", "integrations.slack.slack_app:app", "--host", "0.0.0.0", "--port", "8080"]
EOF

docker build -f Dockerfile.slack -t bedrock-slack-bot:latest .
```

### Run Docker Container

```bash
docker run -d \
  --name bedrock-slack-bot \
  -p 8080:8080 \
  -e SLACK_BOT_TOKEN='xoxb-your-token' \
  -e SLACK_SIGNING_SECRET='your-secret' \
  -e BEDROCK_API_KEY='your-bedrock-key' \
  -e OPENAI_BASE_URL='https://bedrock-mantle.us-east-1.api.aws/v1' \
  -e MODEL_ID='anthropic.claude-3-sonnet-20240229-v1:0' \
  -e WORKSPACE_ROOT='/app' \
  bedrock-slack-bot:latest
```

### Production Deployment

For production, use a cloud service:

**AWS ECS:**
```bash
# Create task definition, service, etc.
# Configure Application Load Balancer
# Get HTTPS URL
```

**Heroku:**
```bash
heroku create your-app-name
heroku config:set SLACK_BOT_TOKEN='xoxb-...'
heroku config:set SLACK_SIGNING_SECRET='...'
git push heroku main
```

**Railway, Render, or other PaaS:**
```bash
# Similar to Heroku with environment variables
```

---

## 🧪 Testing

### Test in Slack

1. Open your Slack workspace
2. Try the slash command:
   ```
   /agent What is 2+2?
   ```
3. Try mentioning the bot in a channel:
   ```
   @Bedrock AI Agent What is Python?
   ```

### Test Endpoints

```bash
# Health check
curl http://localhost:8080/health

# Status
curl http://localhost:8080/admin/status

# Reset conversation
curl -X POST http://localhost:8080/admin/reset-conversation
```

### View Logs

**Terminal:**
```bash
# Already visible in running terminal
```

**Docker:**
```bash
docker logs -f bedrock-slack-bot
```

---

## 🔒 Security Checklist

- ✅ Store tokens in `.env` or secrets manager, never commit
- ✅ Enable signature verification (automatic in this app)
- ✅ Use HTTPS in production
- ✅ Limit file access with `WORKSPACE_ROOT`
- ✅ Add rate limiting if needed
- ✅ Monitor API usage and costs
- ✅ Use Slack app distribution controls
- ✅ Review bot permissions periodically

---

## 📊 Features Supported

### Slash Commands
```
/agent What is Python?
/agent Review my code
/agent Search for React patterns
/agent Handoff to coder
```

### App Mentions
```
@Bedrock AI Agent explain this function
@Bedrock AI Agent find bugs in my code
@Bedrock AI Agent generate documentation
```

### Capabilities
- ✅ File reading from workspace
- ✅ Web search (DuckDuckGo)
- ✅ Weather information
- ✅ Current time
- ✅ Agent handoff to specialists
- ✅ Threaded responses
- ✅ Conversation context
- ✅ Tool execution

---

## 🐛 Troubleshooting

### "Invalid Slack request signature"
- Check `SLACK_SIGNING_SECRET` is correct
- Verify timestamp is within 5 minutes
- Make sure request body isn't modified

### "Slack app not responding"
- Check integration server is running: `curl http://localhost:8080/health`
- Verify Request URL is correct in Slack API settings
- Check logs for errors

### "No response to slash command"
- Ensure `/agent` command is registered in Slack app
- Check request URL format (with `https://`)
- Verify bot token scopes include `commands`

### "Bot not responding to mentions"
- Ensure app is installed to workspace
- Add bot to channel: `@Bedrock AI Agent`
- Check event subscription for `app_mention`

### "Bedrock API error"
- Verify `BEDROCK_API_KEY` is correct
- Check AWS Bedrock service availability
- Review token limits

---

## 📚 Environment Variables Reference

| Variable | Required | Example |
|----------|----------|---------|
| `SLACK_BOT_TOKEN` | ✅ Yes | `xoxb-123456789...` |
| `SLACK_SIGNING_SECRET` | ✅ Yes | `abcd1234efgh5678...` |
| `BEDROCK_API_KEY` | ✅ Yes | `your-aws-key` |
| `OPENAI_BASE_URL` | ✅ Yes | `https://bedrock-mantle.us-east-1.api.aws/v1` |
| `MODEL_ID` | ✅ Yes | `anthropic.claude-3-sonnet-20240229-v1:0` |
| `WORKSPACE_ROOT` | ⚠️ Recommended | `/home/user/bed_rock` |
| `SLACK_INTEGRATION_PORT` | ❌ Optional | `8080` |
| `MAX_FILE_SIZE` | ❌ Optional | `100` |
| `ALLOW_SYSTEM_FILE_READ` | ❌ Optional | `0` |

---

## 🎓 Usage Examples

### Example 1: Code Review
```
User: /agent Review main.py for bugs and improvements
Bot: 🤖 Processing your request...
     (reads main.py, analyzes with coder agent)
     Here are the issues I found:
     1. Missing error handling...
     2. Performance issue...
```

### Example 2: Web Search
```
User: /agent What are the latest Python best practices?
Bot: 🤖 Processing your request...
     (searches web)
     Based on recent sources:
     - Type hints are now standard
     - Async/await patterns...
```

### Example 3: Multi-step Workflow
```
User: @Bedrock AI Agent Handoff to coder and create a Python API template
Bot: ✅ Switched to coder agent
     Here's a FastAPI template...
```

---

## 📞 Support

- **Slack SDK Issues:** https://github.com/slackapi/python-slack-sdk
- **Bedrock Issues:** AWS Support Console
- **Integration Issues:** Check logs and error messages

---

*Setup Guide Version 1.0*  
*Last Updated: May 18, 2026*

