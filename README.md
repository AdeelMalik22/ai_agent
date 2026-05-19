# 🤖 Bedrock OpenAI-Compatible Tool Agent

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![AWS Bedrock](https://img.shields.io/badge/AWS-Bedrock-FF9900)](https://aws.amazon.com/bedrock/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.136%2B-009688)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.56%2B-FF4B4B)](https://streamlit.io/)

A **production-ready AI agent system** with multiple specialist agents, enforced safety guardrails, and powerful tool execution. Built for developers who need intelligent automation with full control.

**Key Features:**
- 🎯 **Multi-Agent System** – Route tasks to specialist agents (planner, coder, reviewer, general)
- 🛡️ **Safety First** – Input/output guardrails with hateful content detection
- 🔧 **Tool Execution** – File I/O, web search, weather, current time, agent handoff
- 🚀 **Multiple Interfaces** – Terminal CLI, Streamlit Web UI, FastAPI REST API
- 🐳 **Docker Ready** – Full Docker & Docker Compose support
- 📊 **AWS Bedrock Integration** – OpenAI-compatible API endpoint support
- 💾 **Streaming Responses** – Real-time streaming of AI responses

**Perfect for:** Code reviews, document analysis, research automation, AI-powered workflow integration, and team collaboration.

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
┌──────────────────────────────────────────────────────────┐
│      User Input (CLI Terminal / Web UI / REST API)       │
└─────────────────────────┬────────────────────────────────┘
                          │
                          ▼
        ┌──────────────────────────────────┐
        │  Input Validation & Guardrails   │
        │  • Prompt injection detection    │
        │  • Token limits enforcement      │
        │  • Hateful content blocking      │
        │  • Repeated message detection    │
        └──────────────┬───────────────────┘
                       │
                       ▼
        ┌──────────────────────────────────┐
        │  Route to Appropriate Agent      │
        │  (determine best specialist)     │
        └──────────────┬───────────────────┘
                       │
        ┌──────────────┴──────────────────┐
        │                                 │
        ▼                                 ▼
    ┌──────────┐                  ┌────────────────┐
    │  General │◄────────────────►│ Specialist     │
    │  Agent   │  (handoff)       │ Agents:        │
    └────┬─────┘                  │ • Planner      │
         │                        │ • Coder        │
         │                        │ • Reviewer     │
         │                        └────────┬───────┘
         │                                 │
         └─────────────┬───────────────────┘
                       │
                       ▼
        ┌──────────────────────────────────┐
        │     Tool Execution Engine        │
        ├──────────────────────────────────┤
        │ • read_file (workspace/system)   │
        │ • write_file (workspace only)    │
        │ • list_files (directory listing) │
        │ • web_search (DuckDuckGo)        │
        │ • get_weather (Open-Meteo API)   │
        │ • get_current_time (UTC)         │
        │ • handoff_to_agent (routing)     │
        └────────────┬─────────────────────┘
                     │
                     ▼
        ┌──────────────────────────────────┐
        │  Output Safety Guardrails        │
        │  • Length enforcement            │
        │  • Hateful content filtering     │
        │  • Response validation           │
        └────────────┬─────────────────────┘
                     │
                     ▼
        ┌──────────────────────────────────┐
        │   Stream Response to User        │
        │   (CLI/UI/API with real-time)    │
        └──────────────────────────────────┘
```

### Data Flow Highlights

1. **Input Security** – Every user prompt is validated against injection patterns, token limits, and content policies before reaching the AI
2. **Agent Routing** – Messages are intelligently routed to specialist agents based on task type
3. **Tool Execution** – Agents can invoke tools with safety checks and argument validation
4. **Output Filtering** – Responses are filtered for harmful content and token limits
5. **Streaming** – Real-time token-by-token streaming for responsive UX

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
├── config/                          # Configuration management
│   ├── __init__.py
│   └── settings.py                  # Environment variables & defaults
│
├── core/                            # Core system components
│   ├── __init__.py
│   ├── session.py                   # Session initialization
│   └── streaming.py                 # Token-by-token streaming
│
├── guardrails/                      # Safety & validation layer
│   ├── __init__.py
│   ├── input_guardrils.py          # User input validation & filtering
│   └── output_guardrils.py         # Response safety filtering
│
├── server/                          # REST API server
│   ├── __init__.py
│   └── api_server.py               # FastAPI application (port 3000/8000)
│
├── utils/                           # Utility functions
│   ├── __init__.py
│   └── tooling.py                  # Tool execution & validation
│
├── documentation/                   # Extended documentation
│   ├── EXTENSION_INTEGRATION.md    # VS Code integration
│   ├── FILE_TOOLS.md               # File I/O documentation
│   ├── GUARDRILS_INPUT_OUTPUT_FLOW.md  # Guardrails deep dive
│   ├── HANDOFF_FLOW.md             # Agent routing logic
│   └── NEW_FEATURES.md             # Advanced features guide
│
├── __pycache__/                     # Python bytecode cache
├── tools.py                         # Tool implementations & schemas
├── system_prompt.py                 # Agent prompts & personalities
├── ai_agents.py                     # CLI entry point
├── streamlit_app.py                 # Streamlit Web UI entry point
├── Dockerfile                       # Single container image
├── Dockerfile.streamlit             # Streamlit container image
├── docker-compose.yaml              # Multi-container orchestration
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

### Component Responsibilities

| Component | Purpose | Key Files |
|-----------|---------|-----------|
| **Config** | Load & validate environment settings | `settings.py` |
| **Core** | Session management, streaming logic | `session.py`, `streaming.py` |
| **Guardrails** | Input validation, output filtering | `input_guardrils.py`, `output_guardrils.py` |
| **Tools** | Function definitions & implementations | `tools.py`, `utils/tooling.py` |
| **Server** | FastAPI REST API | `server/api_server.py` |
| **UI** | Streamlit web interface | `streamlit_app.py` |
| **CLI** | Terminal interface | `ai_agents.py` |

### Execution Flow

```
User Input
    ↓
validate_user_input() → GuardrailConfig checks
    ↓
validate_recent_user_repetition() → Check for spam
    ↓
stream_model_response() → AWS Bedrock API call with tools
    ↓
Tool Calls? → Yes → validate_single_tool_call() → execute_single_tool()
    ↓                         ↓
    └─────────────→ Tool Result → Loop back to model
    ↓
    No → guard_assistant_output() → Output safety checks
    ↓
trim_conversation_history() → Enforce max messages
    ↓
Stream to User (CLI/UI/API)
```

---

## 🐳 Docker Deployment

### Single Container (Terminal Mode)

#### Build
```bash
docker build -t bedrock-agent:latest .
```

#### Run
```bash
docker run -it \
  -e BEDROCK_API_KEY='your-key' \
  -e OPENAI_BASE_URL='https://bedrock-mantle.us-east-1.api.aws/v1' \
  -e MODEL_ID='anthropic.claude-3-sonnet-20240229-v1:0' \
  -v ~/bed_rock:/workspace \
  bedrock-agent:latest
```

### Web UI (Streamlit Container)

Build with dedicated Dockerfile:
```bash
docker build -f Dockerfile.streamlit -t bedrock-streamlit:latest .
```

Run on port 8501:
```bash
docker run -it \
  -e BEDROCK_API_KEY='your-key' \
  -p 8501:8501 \
  -v ~/bed_rock:/app \
  bedrock-streamlit:latest
```

Open: http://localhost:8501

### API Server (FastAPI Container)

Run FastAPI on port 8000:
```bash
docker run -it \
  -e BEDROCK_API_KEY='your-key' \
  -p 8000:8000 \
  -v ~/bed_rock:/app \
  bedrock-agent:latest \
  uvicorn server.api_server:app --host 0.0.0.0 --port 8000
```

Test: `curl http://localhost:8000/health`

### Docker Compose (Full Stack)

Run all services together:
```bash
docker-compose up -d
```

Services:
- **Backend API:** http://localhost:8000
- **Streamlit UI:** http://localhost:8501

View logs:
```bash
docker-compose logs -f backend    # API logs
docker-compose logs -f streamlit  # UI logs
```

Stop all:
```bash
docker-compose down
```

### Environment Variables in Docker

Create `.env` file:
```bash
BEDROCK_API_KEY=your-key
OPENAI_BASE_URL=https://bedrock-mantle.us-east-1.api.aws/v1
MODEL_ID=anthropic.claude-3-sonnet-20240229-v1:0
WORKSPACE_ROOT=/workspace
MAX_FILE_SIZE=100
```

Docker will automatically load from `.env`:
```bash
docker-compose --env-file .env up
```

### Environment Variables (Complete Reference)

#### AWS Bedrock (Required)
```bash
BEDROCK_API_KEY="your-aws-bedrock-key"
OPENAI_BASE_URL="https://bedrock-mantle.us-east-1.api.aws/v1"
MODEL_ID="anthropic.claude-3-sonnet-20240229-v1:0"  # Or other supported model
```

#### Workspace & File Access
```bash
WORKSPACE_ROOT="/home/username/bed_rock"            # Base directory for file operations
MAX_FILE_SIZE="100"                                 # Max file size in KB (default: 100)
ALLOW_SYSTEM_FILE_READ="0"                          # Allow reading outside workspace (0=no, 1=yes)
ALLOWED_READ_EXTENSIONS=".py,.ts,.js,.json,.md,..." # Comma-separated file types
ALLOWED_WRITE_EXTENSIONS=".py,.ts,.js,.json,.md,..."# Comma-separated file types
```

#### Input Guardrails
```bash
MAX_USER_INPUT_LENGTH="4000"                        # Max chars per message
MAX_TOOL_ARGUMENTS_LENGTH="4000"                    # Max chars for tool args
MAX_HANDOFF_REASON_LENGTH="300"                     # Max chars for handoff reason
REPEAT_MESSAGE_WINDOW="6"                           # Check last N messages for spam
REPEAT_MESSAGE_MAX_COUNT="2"                        # Max repetitions allowed
BLOCK_HATEFUL_INPUT="1"                             # Block hateful prompts (0=no, 1=yes)
```

#### Output Guardrails
```bash
MAX_ASSISTANT_OUTPUT_LENGTH="500000"                # Max chars for AI responses
MAX_TOOL_OUTPUT_LENGTH="100000"                     # Max chars for tool results
BLOCK_HATEFUL_OUTPUT="1"                            # Block hateful responses (0=no, 1=yes)
EMPTY_ASSISTANT_FALLBACK="..."                      # Response if output blocked
EMPTY_TOOL_FALLBACK="..."                           # Response if tool output blocked
```

#### Agent Behavior
```bash
MAX_TOOL_ITERATIONS="800"                           # Max tool calls per user message
MAX_HANDOFFS_PER_TURN="2"                           # Max agent handoffs per turn
DEBUG_HANDOFFS="1"                                  # Print handoff debug info (0=no, 1=yes)
DEFAULT_AGENT="general"                             # Default active agent
STREAM_DELAY="0.05"                                 # Delay between stream chunks (seconds)
```

#### JSON Validation
```bash
MAX_JSON_KEYS="50"                                  # Max keys in JSON objects
MAX_JSON_DEPTH="5"                                  # Max nesting depth
MAX_JSON_LIST_ITEMS="50"                            # Max items in JSON arrays
```

#### Streamlit UI (Optional)
```bash
STREAMLIT_SERVER_PORT="8501"
STREAMLIT_SERVER_HEADLESS="true"
STREAMLIT_CLIENT_SHOW_ERROR_DETAILS="false"
```

### Configuration Methods

#### 1. Environment Variables (Highest Priority)
```bash
export BEDROCK_API_KEY='your-key'
export MODEL_ID='anthropic.claude-3-sonnet-20240229-v1:0'
export WORKSPACE_ROOT='/home/user/project'
python ai_agents.py
```

#### 2. .env File (Project Root)
```bash
# .env
BEDROCK_API_KEY=your-key
OPENAI_BASE_URL=https://bedrock-mantle.us-east-1.api.aws/v1
MODEL_ID=anthropic.claude-3-sonnet-20240229-v1:0
WORKSPACE_ROOT=/home/user/bed_rock
MAX_FILE_SIZE=100
ALLOW_SYSTEM_FILE_READ=0
```

Then load with:
```bash
python -m dotenv run ai_agents.py
```

#### 3. Programmatic Configuration (Python)
```python
import os
from config.settings import load_config

os.environ['BEDROCK_API_KEY'] = 'your-key'
os.environ['MAX_USER_INPUT_LENGTH'] = '5000'

guardrail_config, output_guardrail_config, runtime_config = load_config()
```

---

## 🛠️ Available Tools

The agent automatically calls these tools to accomplish tasks. All tools are validated for security before execution.

### File Operations
| Tool | Description | Parameters | Returns |
|------|-------------|------------|---------|
| **`read_file`** | Read file contents (text/code) | `path` (str), `max_lines` (int, opt) | File content or error |
| **`write_file`** | Create or overwrite files | `path` (str), `content` (str) | Success message or error |
| **`list_files`** | List directory contents | `path` (str) | List of files/folders |

**Security Notes:**
- File paths restricted to `WORKSPACE_ROOT` by default
- Enable `ALLOW_SYSTEM_FILE_READ` to read system files
- Only specific file extensions allowed (`.py`, `.js`, `.md`, `.json`, etc.)
- File size limited to `MAX_FILE_SIZE` (default: 100 KB)

### Information & Search
| Tool | Description | Parameters | Returns |
|------|-------------|------------|---------|
| **`web_search`** | Search the web (DuckDuckGo) | `query` (str), `max_results` (int, opt) | Results with title, body, URL |
| **`get_weather`** | Get current weather | `city` (str), `unit` (str, opt) | Temperature, conditions, source |
| **`get_current_time`** | Get current time | None | UTC timestamp |

**Data Sources:**
- **Web Search:** DuckDuckGo (no API key required, privacy-focused)
- **Weather:** Open-Meteo API (geocoding via OpenStreetMap Nominatim)
- **Time:** System UTC time with timezone support

### Agent Routing
| Tool | Description | Parameters | Returns |
|------|-------------|------------|---------|
| **`handoff_to_agent`** | Switch to specialist agent | `target_agent` (str), `reason` (str) | Confirmation of handoff |

**Available Agents:**
- `general` – Default all-purpose agent
- `planner` – Architecture & step-by-step planning
- `coder` – Code implementation & refactoring
- `reviewer` – Code review & bug detection

### Tool Execution Flow
```
Tool Request (from model)
    ↓
validate_single_tool_call()
    • Check tool exists in TOOLS list
    • Validate all required arguments
    • Check argument lengths (MAX_TOOL_ARGUMENTS_LENGTH)
    • Check JSON structure if applicable
    ↓
execute_single_tool()
    • Security checks (file paths, extensions)
    • Execute tool function
    • Capture result
    • Guard output (length, hateful content)
    ↓
Append to conversation as "tool" role message
    ↓
Continue loop (model can make more tool calls or respond to user)
```

---

## 🐛 Troubleshooting

### Setup Issues

#### ❌ "BEDROCK_API_KEY not found"
```
Error: BEDROCK_API_KEY environment variable not set
```
**Solutions:**
```bash
# Option 1: Export as environment variable
export BEDROCK_API_KEY='your-key-here'
source ~/.bashrc  # Reload

# Option 2: Add to .env file
echo "BEDROCK_API_KEY=your-key" >> .env

# Option 3: Verify it's loaded
python -c "import os; print('Key found!' if os.getenv('BEDROCK_API_KEY') else 'Key not found')"
```

#### ❌ "openai.AuthenticationError: Invalid API Key"
```
Error: API request failed with status 401
```
**Solutions:**
- Verify API key is correct in AWS Bedrock console
- Check `OPENAI_BASE_URL` is set correctly
- Ensure AWS credentials have Bedrock access

#### ❌ "Connection refused when accessing API"
```
Error: Connection refused: http://0.0.0.0:3000
Error: Connection refused: http://0.0.0.0:8000
```
**Solution:** Start the API server in a separate terminal
```bash
python server/api_server.py    # Terminal 1
# Then in another terminal, use CLI or UI
python ai_agents.py            # Terminal 2
```

### Tool Execution Issues

#### ❌ "Tool execution failed: File not found"
```
Error: File not found at /path/to/file
```
**Solutions:**
- Check file exists: `ls /path/to/file`
- Verify path is relative to `WORKSPACE_ROOT`
- Use absolute path with `ALLOW_SYSTEM_FILE_READ=1` (risky!)
- Check file permissions: `ls -la /path/to/file`

#### ❌ "Tool execution failed: File extension not allowed"
```
Error: .exe files not allowed for read operations
```
**Solution:** Add extension to allowed list
```bash
export ALLOWED_READ_EXTENSIONS=".py,.js,.txt,.exe"
```

#### ❌ "Tool execution failed: File size exceeds limit"
```
Error: File size 250KB exceeds limit 100KB
```
**Solution:** Increase file size limit
```bash
export MAX_FILE_SIZE="500"  # 500 KB
```

### Response Issues

#### ❌ "Agent keeps looping forever"
```
Message: I hit the tool-iteration limit for this request.
```
**Solutions:**
```bash
# Option 1: Increase limit
export MAX_TOOL_ITERATIONS=50

# Option 2: Simplify your request (too many tool calls needed)

# Option 3: Check for infinite loops in tool output
```

#### ❌ "Response seems truncated or cut off"
```
[Response ends abruptly]
```
**Solutions:**
```bash
# Check output length limits
export MAX_ASSISTANT_OUTPUT_LENGTH=1000000  # 1M chars

# Check if conversation history is too long
export MAX_HISTORY_MESSAGES=100

# Try a simpler request with less context needed
```

#### ❌ "Empty response from model"
```
[WARNING] Empty response from model
```
**Solutions:**
- Check API quota in AWS Bedrock console
- Verify model ID exists and is available
- Try a simple test prompt: "Hello"
- Check internet connection

### Performance Issues

#### ❌ "Responses are very slow"
**Solutions:**
```bash
# Reduce streaming delay (less smooth but faster)
export STREAM_DELAY=0.01

# Use a faster model
export MODEL_ID="anthropic.claude-3-haiku-20240307-v1:0"

# Reduce conversation history
export MAX_HISTORY_MESSAGES=20
```

#### ❌ "High API costs"
**Solutions:**
```bash
# Use cheaper model
export MODEL_ID="anthropic.claude-3-haiku..."

# Limit tool iterations
export MAX_TOOL_ITERATIONS=5

# Keep history shorter
export MAX_HISTORY_MESSAGES=30
```

### Guardrail Issues

#### ⚠️ "Input rejected: Contains prompt injection patterns"
```
Error: Input rejected for security reasons
```
**Explanation:** Your prompt matched suspicious patterns (e.g., "ignore previous instructions", "reveal system prompt")

**Solution:** Rephrase your request naturally

#### ⚠️ "Input rejected: Too many repeated messages"
```
Error: Detected potential spam (repeated message)
```
**Explanation:** You sent the same or similar message multiple times recently

**Solution:** Wait a moment or ask something different

#### ⚠️ "Output blocked: Contains hateful content"
```
AI message was blocked by safety guardrails
```
**Explanation:** Response matched hateful content patterns

**Solution:** Try different phrasing or report if this is a false positive

---

## 🔍 Debug Mode

Enable detailed logging for troubleshooting:

```bash
# Terminal mode
export DEBUG_HANDOFFS=1
export DEBUG_GUARDRAILS=1
python ai_agents.py

# Check logs
tail -f debug.log
```

Common debug output:
```
[tools] requested=read_file,web_search
[handoff] -> planner
[guardrail] input validation passed
[tool_result] read_file: 1234 bytes read
```

---

## 📞 Getting Help

1. **Check Documentation** – See `/documentation` folder for detailed guides
2. **Review Examples** – Try example queries from Usage Examples section
3. **Check Logs** – Enable debug mode to see detailed execution trace
4. **Verify Setup** – Run quick setup test: `python -c "from config.settings import load_config; load_config()"`
5. **Report Issues** – Include error message, environment variables (except API key), and steps to reproduce

---

## 🎓 Usage Examples

### Example 1: Code Review (Terminal)

```bash
$ python ai_agents.py
ask Question.....: Analyze this Python function for bugs and improvements

# Agent will:
# 1. Ask for file path
# 2. Read the file
# 3. Handoff to 'reviewer' agent
# 4. Perform detailed code review
# 5. Suggest improvements
```

### Example 2: Web Research (Web UI)

```bash
1. Start: streamlit run streamlit_app.py
2. Open: http://localhost:8501
3. Ask: "What are the latest developments in AI agents?"
4. Agent will web_search and summarize findings
```

### Example 3: File Analysis & Modification (API)

```python
import requests
import json

# Start API server first: python server/api_server.py

response = requests.post(
    "http://localhost:3000/chat",
    json={
        "message": "Read config.json, summarize the settings, then create a backup with .bak extension",
        "context": {"workspace_root": "/home/user/project"}
    }
)

data = response.json()
print(f"Response: {data['response']}")
print(f"Tools used: {data.get('tools_used', [])}")
```

### Example 4: Multi-Agent Planning (Terminal)

```bash
ask Question.....: I need to build a Python CLI tool for data processing. Create a plan with implementation steps.

# Agent flow:
# 1. Handoff to 'planner' agent
# 2. Generate step-by-step plan
# 3. When you ask "implement step 1", handoff to 'coder' agent
# 4. Coder generates implementation
# 5. When you ask "review the code", handoff to 'reviewer' agent
```

### Example 5: Batch File Processing (API)

```python
import requests

BASE_URL = "http://localhost:3000"

files_to_review = ["main.py", "utils.py", "config.py"]

for file in files_to_review:
    response = requests.post(
        f"{BASE_URL}/chat",
        json={
            "message": f"Review {file} and list 3 main improvements",
            "context": {"workspace_root": "/home/user/project"}
        }
    )
    
    result = response.json()
    print(f"\n{file}:")
    print(result['response'])
```

### Example 6: Tool Chain (Terminal)

```bash
ask Question.....: 
1. Search for "latest Python best practices 2024"
2. Then read my code at main.py
3. Compare and suggest improvements

# Agent automatically:
# 1. Calls web_search tool
# 2. Calls read_file tool  
# 3. Analyzes and provides recommendations
# 4. All with security validation at each step
```

---

## 🧩 Extending the Agent

### Add a New Tool

1. **Define tool schema in `tools.py`:**
```python
TOOLS = [
    # ... existing tools ...
    {
        "type": "function",
        "function": {
            "name": "my_custom_tool",
            "description": "Does something useful",
            "parameters": {
                "type": "object",
                "properties": {
                    "input_param": {
                        "type": "string",
                        "description": "What this parameter does"
                    }
                },
                "required": ["input_param"]
            }
        }
    }
]
```

2. **Implement the function in `tools.py`:**
```python
def my_custom_tool(input_param: str) -> dict:
    """Implementation of my_custom_tool."""
    try:
        result = do_something_with(input_param)
        return {
            "status": "success",
            "result": result,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "message": f"Failed to execute: {str(e)}"
        }
```

3. **Add execution logic in `utils/tooling.py`:**
```python
def execute_single_tool(tool_call, ...):
    # ...existing code...
    if function_name == "my_custom_tool":
        result = my_custom_tool(parsed_args.get("input_param"))
        return json.dumps(result), active_agent, handoffs_this_turn
```

4. **Validate tool inputs (optional but recommended):**
```python
# In guardrails/input_guardrils.py or utils/tooling.py
def validate_my_custom_tool(params: dict) -> tuple[bool, str]:
    if len(params.get("input_param", "")) > 10000:
        return False, "input_param must be under 10000 characters"
    return True, ""
```

### Add a New Specialist Agent

1. **Add agent prompt to `system_prompt.py`:**
```python
AGENT_PROMPTS = {
    "general": "...",
    # ... other agents ...
    "my_specialist": """You are a specialized AI agent for [domain].
    
Your role: [description of what you do]

When to use you: [when the user needs this specialist]

Your capabilities:
- [capability 1]
- [capability 2]
- [capability 3]

Important guidelines:
- [guideline 1]
- [guideline 2]
"""
}
```

2. **Update routing logic in `ai_agents.py`:**
```python
# In process_model_response() or similar location
if "handoff_to_agent" in tool_calls:
    target_agent = tool_call_args.get("target_agent")
    if target_agent == "my_specialist":
        messages[0] = {"role": "system", "content": AGENT_PROMPTS["my_specialist"]}
        active_agent = "my_specialist"
```

3. **Update general agent routing in `system_prompt.py`:**
```python
"general": SYSTEM_PROMPT + """
Routing policy:
- If user asks for [domain task], call handoff_to_agent with target_agent='my_specialist'
- ...other routing...
"""
```

### Modify Guardrails

**Input Guardrails** (`guardrails/input_guardrils.py`):
```python
# Adjust or add blocked phrases
_BLOCKED_PHRASES = (
    "ignore previous instructions",
    # Add your custom blocked phrases
    "my_blocked_phrase",
)

# Adjust or add protected groups
_PROTECTED_GROUPS = (
    "race",
    # Add groups you want to protect
    "profession",
)
```

**Output Guardrails** (`guardrails/output_guardrils.py`):
```python
# Modify output filtering logic
def guard_assistant_output(content: str, config: OutputGuardrailConfig) -> str:
    # Add custom filtering rules
    if "my_unsafe_pattern" in content.lower():
        return config.empty_assistant_fallback
    return content
```

**Configuration** (`config/settings.py`):
```python
# Adjust limits
def load_config():
    guardrail_config = GuardrailConfig(
        max_user_input_length=int(os.getenv("MAX_USER_INPUT_LENGTH", "5000")),  # Increased
        # ... other configs ...
    )
```

---

---

## 📊 Performance & Cost Optimization

### Speed Optimization
```bash
# Reduce token streaming delay (faster but less smooth visual effect)
export STREAM_DELAY=0.01

# Use faster models
export MODEL_ID="anthropic.claude-3-haiku-20240307-v1:0"

# Reduce conversation history
export MAX_HISTORY_MESSAGES=20

# Limit tool iterations
export MAX_TOOL_ITERATIONS=10

# For development: Reduce max output length
export MAX_ASSISTANT_OUTPUT_LENGTH=50000
```

### Cost Optimization
```bash
# Use cheaper models
export MODEL_ID="anthropic.claude-3-haiku..."

# Limit tool calls (each tool call costs money)
export MAX_TOOL_ITERATIONS=5

# Keep history short (fewer tokens per request)
export MAX_HISTORY_MESSAGES=30

# Keep files small (smaller files = fewer tokens)
export MAX_FILE_SIZE=50
```

### Accuracy Optimization
```bash
# Use larger models
export MODEL_ID="anthropic.claude-3-opus-20240229-v1:0"

# Increase available context
export MAX_HISTORY_MESSAGES=100

# Allow more tool iterations for complex problems
export MAX_TOOL_ITERATIONS=50
```

### Benchmarking

Test your setup:
```bash
# Quick response time test
python -c "
import time
from openai import OpenAI
import os

client = OpenAI(
    api_key=os.getenv('BEDROCK_API_KEY'),
    base_url=os.getenv('OPENAI_BASE_URL')
)

start = time.time()
response = client.chat.completions.create(
    model=os.getenv('MODEL_ID'),
    messages=[{'role': 'user', 'content': 'Say hello'}],
    max_tokens=10
)
elapsed = time.time() - start
print(f'Response time: {elapsed:.2f}s')
print(f'Content: {response.choices[0].message.content}')
"
```

---

## ❓ FAQ

**Q: Can I use OpenAI or other providers instead of Bedrock?**  
A: The system is built for AWS Bedrock's OpenAI-compatible endpoint. To use other providers, modify `build_client()` in `ai_agents.py` and `streamlit_app.py` to use their endpoints and adjust authentication accordingly.

**Q: How much does this cost?**  
A: Costs depend on your AWS Bedrock pricing plan:
- Pay-per-token pricing for on-demand
- Commitment pricing for volume discounts
- Monitor API calls in AWS Bedrock console

**Q: Can I run this in production?**  
A: Yes! Use the FastAPI server (`server/api_server.py`):
- Add authentication layer (API keys, JWT, etc.)
- Enable rate limiting (use `slowapi` or similar)
- Add monitoring/logging (CloudWatch, ELK, etc.)
- Use reverse proxy (nginx, load balancer)
- Deploy to Kubernetes, ECS, or serverless

**Q: How do I clear the conversation history?**  
A: 
- **Terminal:** Type `exit` or `quit` to end session
- **Web UI:** Click "Reset Conversation" button in sidebar
- **API:** Each session starts fresh; no persistent conversation

**Q: Can I save and replay conversations?**  
A: Conversations are in-memory only. To save:
```python
import json
# Export messages
with open("conversation.json", "w") as f:
    json.dump(messages, f, indent=2)
# Later, load and use
with open("conversation.json", "r") as f:
    messages = json.load(f)
```

**Q: Is this secure? What about data privacy?**  
A: Security features:
- ✅ Input/output guardrails prevent injection attacks & toxic content
- ✅ File access restricted to workspace by default
- ✅ Configurable file extension whitelist
- ✅ File size limits prevent large data exfiltration
- ⚠️ API calls sent to AWS Bedrock (check AWS privacy policy)
- ⚠️ Web search queries sent to DuckDuckGo
- 🔐 Recommendation: Use in private networks for sensitive data

**Q: Can I use custom system prompts?**  
A: Yes! Edit `system_prompt.py`:
```python
AGENT_PROMPTS = {
    "general": "Your custom system prompt here...",
    # ... other agents ...
}
```

**Q: What models are supported?**  
A: Any model available on AWS Bedrock that supports:
- Tool/function calling
- OpenAI-compatible API
- Streaming

Popular options:
- Claude 3 Opus (best, most capable)
- Claude 3 Sonnet (balanced)
- Claude 3 Haiku (fast, cheap)

**Q: How do I debug issues?**  
A: Enable debug mode:
```bash
export DEBUG_HANDOFFS=1
python ai_agents.py 2>&1 | tee debug.log
```

**Q: Can I integrate with VS Code?**  
A: Yes! See `documentation/EXTENSION_INTEGRATION.md` for integration with VS Code extensions.

---

## 📚 Complete Documentation

### Core Documentation
- **README.md** (this file) – Overview, quick start, and reference
- **`/documentation/NEW_FEATURES.md`** – Advanced features, Docker deep dive, benchmarking
- **`/documentation/GUARDRILS_INPUT_OUTPUT_FLOW.md`** – Detailed safety system explanation
- **`/documentation/HANDOFF_FLOW.md`** – Agent routing and specialist agent logic
- **`/documentation/FILE_TOOLS.md`** – File I/O tools detailed documentation
- **`/documentation/EXTENSION_INTEGRATION.md`** – VS Code and IDE integration

### Source Code Documentation

#### Entry Points
- **`ai_agents.py`** – CLI terminal interface (start here: `python ai_agents.py`)
- **`streamlit_app.py`** – Web UI interface (start here: `streamlit run streamlit_app.py`)
- **`server/api_server.py`** – FastAPI REST API (start here: `python server/api_server.py`)

#### Core Components
- **`config/settings.py`** – Configuration loader and environment management
- **`core/session.py`** – Session initialization with agents and tools
- **`core/streaming.py`** – Token-by-token streaming from AWS Bedrock
- **`tools.py`** – Tool definitions, schemas, and implementations
- **`system_prompt.py`** – Agent system prompts and personalities

#### Safety & Validation
- **`guardrails/input_guardrils.py`** – Input validation and filtering logic
- **`guardrails/output_guardrils.py`** – Output safety and content filtering
- **`utils/tooling.py`** – Tool execution, validation, and error handling

#### Configuration
- **`requirements.txt`** – Python dependencies (pip install -r requirements.txt)
- **`Dockerfile`** – Single-container image
- **`Dockerfile.streamlit`** – Web UI container image
- **`docker-compose.yaml`** – Multi-container orchestration

---

## 🚀 Next Steps

### For First-Time Users
1. ✅ Follow **Quick Start** section above (2 minutes)
2. 📖 Try **Usage Examples** (5 minutes)
3. 🎮 Explore tools: Ask agent to read files, search web, etc.
4. 🔄 Try switching agents: "Handoff to coder" or ask planning questions

### For Integration
1. 📘 Read **Docker Deployment** section for containerization
2. 🔌 Review **API Integration** example to connect with your systems
3. 🛡️ Understand **Guardrails** for security considerations
4. ⚙️ Adjust **Configuration** for your use case

### For Extension
1. 🧩 Follow **Extending the Agent** section to add tools
2. 👥 Create new **Specialist Agents** for your domain
3. 🛡️ Customize **Guardrails** for your safety requirements
4. 📊 Implement **Benchmarking** to measure improvements

### For Production
1. 🐳 Use **Docker Compose** for reliability
2. 🔐 Add authentication layer to API
3. 📈 Enable monitoring and logging
4. 💾 Implement conversation persistence (optional)
5. ⚡ Optimize performance based on **Performance Tips**
6. 💰 Monitor costs and optimize model selection

---

## 🤝 Contributing & Support

### Issues & Feedback
- 📝 Report bugs with error messages and reproduction steps
- 💡 Suggest features with use cases and benefits
- 📸 Include debug logs (with sensitive data removed)

### Development Setup
```bash
git clone <repo>
cd bed_rock
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Make changes
python ai_agents.py  # Test CLI
streamlit run streamlit_app.py  # Test Web UI

# Create PR with clear description
```

### Code Standards
- Follow PEP 8 for Python style
- Add docstrings to functions and classes
- Include type hints where practical
- Test changes before submitting
- Update documentation for new features

---

## 📄 License & Attribution

**License:** MIT License – See LICENSE file for full text

**Key Dependencies:**
- OpenAI Python SDK – OpenAI API client
- FastAPI – Web framework
- Streamlit – UI framework
- Pydantic – Data validation
- DDGS (DuckDuckGo Search) – Web search
- AWS Bedrock SDK – Model access

---

*Last Updated: May 18, 2026*  
*Version: 2.0 (Comprehensive Edition)*  
*Maintained by: Bedrock Agent Team*

---

## 🎯 Common Starting Points

**I want to...**
- ✅ **Use the CLI** → Run `python ai_agents.py`
- ✅ **Use the Web UI** → Run `streamlit run streamlit_app.py`
- ✅ **Build an API** → Run `python server/api_server.py` and check `/health`
- ✅ **Use Docker** → See **Docker Deployment** section
- ✅ **Review code** → Ask agent to "review [filename]"
- ✅ **Plan a project** → Handoff to planner agent
- ✅ **Generate code** → Handoff to coder agent
- ✅ **Search the web** → Agent can call `web_search` tool automatically
- ✅ **Read files** → Agent can call `read_file` tool automatically
- ✅ **Add a new tool** → See **Extending the Agent** section
- ✅ **Debug issues** → See **Troubleshooting** section
- ✅ **Understand architecture** → See **Architecture Overview** section

**Questions?** → Check FAQ or relevant documentation section

