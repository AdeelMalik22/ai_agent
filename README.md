# Bedrock OpenAI‑Compatible Tool Agent

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An intelligent AI assistant that can be used via **CLI**, **Streamlit UI**, or **FastAPI API**. It runs on AWS Bedrock (OpenAI-compatible endpoint), features multiple specialist agents, enforces input/output guardrails, and executes local tools (file I/O, web search, agent handoff).

**Perfect for:** Code reviews, document analysis, research workflows, and AI-powered automation.

---

## 🚀 Quick Start (2 Minutes)

### 1️⃣ Clone & Install
```bash
git clone <repo>
cd bed_rock
python -m venv .venv
source .venv/bin/activate          # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2️⃣ Configure Environment
```bash
# Add to ~/.bashrc, ~/.zshrc, or .env file
export BEDROCK_API_KEY='your-bedrock-key-here'
export OPENAI_BASE_URL='https://bedrock-mantle.us-east-1.api.aws/v1'
export MODEL_ID='anthropic.claude-3-sonnet-20240229-v1:0'
export WORKSPACE_ROOT='/home/username/bed_rock'
```

### 3️⃣ Run
```bash
# Terminal (fastest for development)
python ai_agents.py

# Web UI (best UX)
streamlit run streamlit_app.py      # Opens http://localhost:8501

# API Server (production)
python server/api_server.py         # API at http://localhost:3000
```

**That's it!** Start chatting with the AI agent.

---

## 📋 System Requirements

### Required
- **Python:** 3.10 or higher
- **AWS Account:** With Bedrock access enabled
- **Disk Space:** ~500MB for dependencies and virtual environment
- **RAM:** 4GB minimum (8GB recommended)
- **Internet:** Required for API calls and web search

### API Keys
| Key | Source | Required |
|-----|--------|----------|
| `BEDROCK_API_KEY` | AWS Bedrock Console | ✅ Yes |
| `BEDROCK_MODEL_ID` | AWS Bedrock Available Models | ✅ Yes |
| `WORKSPACE_ROOT` | Your project directory | ⚠️ Recommended |

### Tested On
- Ubuntu 22.04 LTS
- macOS 12+ (Intel & Apple Silicon)
- Windows 11 (with WSL2 recommended)

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│         User Input (CLI / Web UI / API)              │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
        ┌────────────────────────────┐
        │  Input Guardrails          │
        │  (validation, limits)       │
        └────────────┬───────────────┘
                     │
                     ▼
        ┌────────────────────────────┐
        │  Route to Agent            │
        │  (general/planner/coder)   │
        └────────────┬───────────────┘
                     │
        ┌────────────┴────────────────┐
        │                             │
        ▼                             ▼
    ┌─────────┐              ┌──────────────┐
    │ General │◄────────────►│ Specialist   │
    │ Agent   │              │ Agents       │
    └────┬────┘              │ - Planner    │
         │                   │ - Coder      │
         │                   │ - Reviewer   │
         │                   └──────┬───────┘
         │                          │
         └──────────┬───────────────┘
                    │
                    ▼
        ┌───────────────────────────┐
        │    Tool Execution         │
        ├───────────────────────────┤
        │ • read_file (any path)    │
        │ • write_file              │
        │ • list_files              │
        │ • get_weather             │
        │ • web_search              │
        │ • get_current_time        │
        └────────────┬──────────────┘
                     │
                     ▼
        ┌───────────────────────────┐
        │  Output Guardrails        │
        │  (safety filters)         │
        └────────────┬──────────────┘
                     │
                     ▼
        ┌───────────────────────────┐
        │   Stream Response         │
        │   (CLI/UI/API)            │
        └───────────────────────────┘
```

---

## 🎯 Which Interface Should You Use?

| Feature | Terminal | Web UI | API |
|---------|----------|--------|-----|
| **Setup Time** | 30s | 30s | 2min |
| **Speed** | ⚡ Fastest | 🟡 Medium | 🟡 Fast |
| **User Experience** | Basic | ✅ Best | Programmatic |
| **History Persistence** | Session only | Session only | Stateful (optional) |
| **Team Sharing** | Local only | Shareable (localhost) | Full integration |
| **Debugging** | ✅ Full logs | Limited | ✅ Full logs |
| **Best For** | Development | Demos | Production |

**Quick Guide:**
- 💻 **Terminal:** You're debugging or doing development
- 🎨 **Web UI:** You want a nice interface and to show demos
- 🔌 **API:** You're integrating into other tools or production services

---

## 📦 Core Architecture

### Directory Structure
```
bed_rock/
├── config/              # Configuration
│   └── settings.py      # Environment & defaults
├── core/                # Core functionality
│   ├── session.py       # Session management
│   └── streaming.py     # Streaming responses
├── guardrails/          # Safety & validation
│   ├── input_guardrils.py   # Input validation
│   └── output_guardrils.py  # Output filtering
├── server/              # API server
│   ├── api_server.py    # FastAPI routes
│   └── main.py          # Uvicorn entry
├── utils/               # Utilities
│   └── tooling.py       # Tool execution
├── documentation/       # Extended docs
├── tools.py             # Tool definitions & implementations
├── system_prompt.py     # Agent system prompts
├── ai_agents.py         # CLI entry point
├── streamlit_app.py     # Web UI entry point
└── requirements.txt     # Dependencies
```

### How It Works

1. **Startup** – Loads environment variables, initializes session, creates OpenAI client
2. **User Input** – Message appended to conversation history
3. **Input Validation** – Checks for length, spam, injection attempts
4. **Model Processing** – Streams response from AWS Bedrock
5. **Tool Calling** – If agent requests tools, validates and executes them
6. **Agent Routing** – Can handoff to specialist agents (planner, coder, reviewer)
7. **Output Safety** – Filters unsafe content before streaming
8. **Response** – Streams to user via CLI, UI, or API

---

## ⚙️ Configuration

### Environment Variables
```bash
# AWS Bedrock (required)
export BEDROCK_API_KEY="your-key-here"
export OPENAI_BASE_URL="https://bedrock-mantle.us-east-1.api.aws/v1"
export MODEL_ID="anthropic.claude-3-sonnet-20240229-v1:0"

# Workspace (recommended)
export WORKSPACE_ROOT="/home/username/bed_rock"

# File I/O (optional)
export MAX_FILE_SIZE="100"          # KB, default 100
export SYSTEM_FILE_ACCESS="true"    # Allow reading system files

# Agent behavior (optional)
export MAX_TOOL_ITERATIONS="8"      # Max tool calls per turn
export ENABLE_METRICS="false"       # Performance metrics

# UI (optional - Streamlit)
export STREAMLIT_SERVER_PORT="8501"
export STREAMLIT_SERVER_HEADLESS="true"
```

### Via .env File
Create a `.env` file in the project root:
```bash
# Copy and fill in your values
BEDROCK_API_KEY=your-key
OPENAI_BASE_URL=https://bedrock-mantle.us-east-1.api.aws/v1
MODEL_ID=anthropic.claude-3-sonnet-20240229-v1:0
WORKSPACE_ROOT=/home/username/bed_rock
```

---

## 🛠️ Available Tools

The agent can use these tools to accomplish tasks:

### File I/O
- **`read_file`** – Read files from your workspace or system
- **`write_file`** – Create/modify files in your workspace
- **`list_files`** – List directory contents

### Information
- **`get_current_time`** – Get current UTC time
- **`get_weather`** – Get weather for a city

### Advanced
- **`web_search`** – Search the web for information
- **`handoff_to_agent`** – Route to specialist agents

---

## 🐛 Troubleshooting

### Common Issues & Solutions

#### ❌ "Bedrock API Key not found"
```
Error: BEDROCK_API_KEY environment variable not set
```
**Solution:**
```bash
export BEDROCK_API_KEY='your-key-here'
# Verify it loaded:
python -c "import os; print(os.getenv('BEDROCK_API_KEY'))"
```

#### ❌ "Connection refused: 0.0.0.0:3000"
```
Error: [Errno 111] Connection refused
```
**Solution:** Make sure the API server is running
```bash
python server/api_server.py    # In a separate terminal
```

#### ❌ "Tool execution failed: File not found"
```
Error: File not found: /some/path
```
**Solution:** 
- Ensure file path is relative to `WORKSPACE_ROOT`
- Or use absolute path with `SYSTEM_FILE_ACCESS=true`
- Check file permissions

#### ❌ "Agent keeps looping and doesn't stop"
```
I hit the tool-iteration limit for this request.
```
**Solution:** Increase tool iteration limit
```bash
export MAX_TOOL_ITERATIONS=12
```

#### ❌ "Bedrock rate limit exceeded"
```
Error: Rate limit exceeded
```
**Solution:** Wait a moment and try again. Consider batch processing requests.

#### ❌ "Model token limit exceeded"
```
Error: This model's maximum context length is 4096 tokens
```
**Solution:** Clear conversation history or use a model with larger context window

---

## 🎓 Usage Examples

### Example 1: Code Review (Terminal)
```bash
$ python ai_agents.py
agent> Review the code in main.py and suggest improvements

[Agent reads main.py, analyzes it, provides detailed review]
```

### Example 2: Document Analysis (Web UI)
```bash
1. Start: streamlit run streamlit_app.py
2. Open: http://localhost:8501
3. Upload or paste document
4. Ask: "Summarize this and extract key points"
```

### Example 3: API Integration (Python)
```python
import requests
import json

response = requests.post(
    "http://localhost:3000/chat",
    json={
        "message": "Read config.json and explain the settings",
        "context": {"workspace_root": "/home/user/project"}
    }
)

print(response.json()["response"])
```

---

## 🧩 Extending the Agent

### Add a New Tool

1. **Define the schema in `tools.py`:**
```python
{
    "type": "function",
    "function": {
        "name": "my_tool",
        "description": "What this tool does",
        "parameters": {
            "type": "object",
            "properties": {
                "param1": {"type": "string", "description": "..."}
            },
            "required": ["param1"]
        }
    }
}
```

2. **Implement the function in `tools.py`:**
```python
def my_tool(param1: str) -> dict:
    # Your implementation
    return {"result": "..."}
```

3. **Add to TOOLS list** and implement in `run_tool()` function

### Add a New Specialist Agent

1. **Add to `AGENT_PROMPTS` in `system_prompt.py`:**
```python
AGENT_PROMPTS = {
    "my_specialist": {
        "system": "You are a specialized agent that...",
        "tools": ["tool1", "tool2"]
    }
}
```

2. **Update routing logic** to handoff to your new agent when needed

### Modify Guardrails

Edit `guardrails/input_guardrils.py` and `guardrails/output_guardrils.py`:
```python
# Adjust limits in config/settings.py or directly in guardrail files
DEFAULT_MAX_USER_INPUT_LENGTH = 8000  # Increase if needed
```

---

## 📊 Performance Tips

- **Speed up responses:** Reduce `MAX_TOOL_ITERATIONS`
- **Better accuracy:** Increase model context or use a larger model
- **Lower costs:** Use smaller model or limit tool calls
- **File access:** Keep files under 100KB for faster reads

---

## ❓ FAQ

**Q: Can I use this with OpenAI directly instead of Bedrock?**  
A: Currently configured for Bedrock. Support for other providers coming soon.

**Q: How much does this cost?**  
A: Depends on your Bedrock plan. Monitor API calls in AWS console.

**Q: Can I run this in production?**  
A: Yes! Use the API server (`server/api_server.py`) with proper error handling and monitoring.

**Q: How do I clear the conversation history?**  
A: Type `clear` in terminal, or refresh the Streamlit page, or start a new session.

**Q: Can I save conversations?**  
A: Conversations are kept in memory during a session. Export feature coming soon.

**Q: Is this secure?**  
A: Input/output guardrails prevent injection attacks. File access is restricted to workspace by default.

---

## 📚 Documentation

Extended documentation available in `/documentation`:
- **`NEW_FEATURES.md`** – 🚀 All new features & advanced setup (Docker, Metrics, Export, Replay, Profiles, Benchmarking, Multi-Model, Cost Tracking, Setup Wizard)
- `EXTENSION_INTEGRATION.md` – VS Code extension integration guide
- `FILE_TOOLS.md` – Detailed file I/O tools documentation
- `GUARDRILS_INPUT_OUTPUT_FLOW.md` – Safety system explanation
- `HANDOFF_FLOW.md` – Agent routing and handoff logic
- `README.md` – Full system documentation

---

## 🚀 Next Steps

1. **Get started** – Follow the Quick Start section above
2. **Explore tools** – Try asking the agent to read files, search web, etc.
3. **Customize** – Add your own tools and specialist agents
4. **Integrate** – Use the API server to integrate with other applications
5. **Contribute** – Report issues or suggest improvements

---

## 📄 License

MIT License – See LICENSE file for details

---

## 🤝 Contributing

Found a bug? Have a feature idea? Please open an issue or pull request!

### Development Setup
```bash
git clone <repo>
cd bed_rock
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# Make your changes
python ai_agents.py  # Test
```

---

## 📞 Support

- 📖 Read the documentation in `/documentation`
- 🐛 Report bugs via GitHub Issues
- 💬 Ask questions in Discussions
- 📧 Email: [contact info if available]

---

*Last updated: April 30, 2024*
*Maintained by the Bedrock Agent team*
