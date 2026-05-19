# 🔌 Integrations

This directory contains integrations for Bedrock AI Agent with external platforms.

---

## 📚 Available Integrations

### 🎯 Slack Integration

**Status:** ✅ Production Ready  
**Directory:** `./slack/`

Integrate Bedrock AI Agent with Slack to:
- Use slash commands (`/agent`)
- Get mentioned in channels (`@Bedrock AI Agent`)
- Use all tools (file read/write, web search, etc.)
- Switch between specialist agents
- Maintain conversation context

#### Quick Start
```bash
# 1. Install dependencies
pip install -r slack/requirements.txt

# 2. Set environment variables
export SLACK_BOT_TOKEN='xoxb-...'
export SLACK_SIGNING_SECRET='...'
export BEDROCK_API_KEY='...'
# ... other required vars

# 3. Start server
python integrations/slack/run.py

# 4. Use ngrok to expose (for testing)
ngrok http 8080

# 5. Update URLs in Slack app settings
```

#### Full Setup
See [slack/SETUP.md](slack/SETUP.md) for complete setup guide with:
- Step-by-step Slack app creation
- Local development setup
- Docker deployment
- Testing and troubleshooting

#### Key Features
✅ Slash commands  
✅ App mentions  
✅ Real-time responses  
✅ Tool execution  
✅ Agent handoff  
✅ Conversation context  
✅ Signature verification  
✅ Request logging  

---

## 🚀 Future Integrations

Planned integrations:
- Discord
- Microsoft Teams
- Telegram
- Matrix/Element
- Twilio SMS
- Email

---

## 🔧 Creating New Integrations

### Template Structure

```
integrations/
├── __init__.py
└── my_service/
    ├── __init__.py
    ├── config.py          # Configuration
    ├── handler.py         # Core logic
    ├── app.py            # API/server
    ├── requirements.txt  # Dependencies
    ├── SETUP.md          # Setup guide
    └── run.py            # Quick start script
```

### Example Implementation

```python
# handler.py
class MyServiceHandler:
    def __init__(self, api_key: str):
        self.client = MyServiceClient(api_key)
        # Initialize Bedrock agent
    
    def handle_message(self, text: str) -> str:
        # 1. Validate input
        # 2. Forward to agent
        # 3. Return response
        pass
```

### Key Principles

1. **Security First**
   - Verify all requests (signatures, tokens, etc.)
   - Never log sensitive data
   - Use environment variables for secrets

2. **Reuse Core**
   - Import from main agent system
   - Use guardrails for validation
   - Leverage existing tools

3. **Consistent Interface**
   - Config class with validation
   - Handler class with message processing
   - FastAPI/Flask app for webhooks
   - SETUP.md for documentation

4. **Production Ready**
   - Error handling throughout
   - Logging and debugging
   - Rate limiting
   - Timeout handling

---

## 📊 Integration Status

| Integration | Status | Features | Cost |
|------------|--------|----------|------|
| Slack | ✅ Ready | Slash, mentions, tools | FREE |
| Discord | 🏗️ Planned | Commands, reactions | FREE |
| Teams | 🏗️ Planned | Adaptive cards | FREE |
| Telegram | 🏗️ Planned | Commands, inline | FREE |
| SMS | 🔜 Future | Twilio | $$$ |
| Email | 🔜 Future | SMTP | FREE |

---

## 📞 Support

For integration-specific help:
- See `<integration>/SETUP.md` for setup
- Check `<integration>/handler.py` for code docs
- Review logs with `<integration>/run.py`

---

*Integrations Directory*  
*Last Updated: May 18, 2026*

