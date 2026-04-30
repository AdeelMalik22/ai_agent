# Bedrock OpenAI-Compatible Tool Agent

An intelligent multi-agent AI assistant system that uses the OpenAI SDK against AWS Bedrock with local function tool-calling, web search, multi-agent routing, and interactive UI support.

## What this project does

- **Multi-Agent System**: Routes requests between specialist agents (general, planner, coder, reviewer)
- **Tool Calling**: Executes function calls locally with web search capabilities
- **Interactive Interfaces**: Terminal CLI, Streamlit web UI, and FastAPI backend
- **Guardrails**: Input validation, output filtering, and safety constraints
- **Streaming Responses**: Real-time response rendering as they're generated

## Project Structure

### Core Components

- **`ai_agents.py`** - Terminal CLI main interactive loop and model orchestration
- **`streamlit_app.py`** - Web UI frontend (browser-based chat interface)
- **`api_server.py`** - FastAPI backend for programmatic access
- **`system_prompt.py`** - Agent personalities and routing policies
- **`tools.py`** - Tool schemas and execution handlers
- **`config/settings.py`** - Configuration and environment loading
- **`core/`** - Session management and streaming utilities
- **`guardrails/`** - Input/output validation and safety constraints
- **`utils/tooling.py`** - Tool utility functions

### Key Files

- `HANDOFF_FLOW.md` - Complete guide to multi-agent handoff mechanism
- `GUARDRILS_INPUT_OUTPUT_FLOW.md` - Safety guardrails documentation

## Prerequisites

- Python 3.10+
- AWS Bedrock access with OpenAI-compatible endpoint
- Installed Python packages (see `requirements.txt`)

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Environment Variables

Add to `~/.bashrc` or `~/.zshrc`:

```bash
export BEDROCK_API_KEY='your-bedrock-api-key'
export OPENAI_API_KEY='your-openai-api-key'  # optional, uses BEDROCK_API_KEY if not set
export OPENAI_BASE_URL='https://bedrock-mantle.us-east-1.api.aws/v1'
export MODEL_ID='openai.gpt-oss-120b'

# Optional configuration
export MAX_USER_INPUT_LENGTH=4000
export MAX_TOOL_ITERATIONS=8
export MAX_HANDOFFS_PER_TURN=2
export DEBUG_HANDOFFS=1
export MAX_HISTORY_MESSAGES=40

# File I/O Tools Configuration
export WORKSPACE_ROOT=/path/to/your/workspace
export MAX_FILE_SIZE=100
export ALLOWED_READ_EXTENSIONS=".py,.ts,.js,.json,.md,.txt,.yaml,.yml,.toml,.env,.sh,.css,.html,.xml,.sql,.r,.rb,.go,.java,.cpp,.c,.h,.cs"
export ALLOWED_WRITE_EXTENSIONS=".py,.ts,.js,.json,.md,.txt,.yaml,.yml,.toml,.env,.sh,.css,.html,.xml,.sql,.r,.rb,.go,.java,.cpp,.c,.h,.cs"
```

Then reload your shell:

```bash
source ~/.bashrc
```

## Running the Application

### Option 1: Terminal CLI

```bash
python ai_agents.py
```

**Usage:**
- Ask questions like: `what time is it?`, `weather in london`, `review this code...`
- Type `exit` or `quit` to stop

### Option 2: Web UI (Streamlit)

```bash
streamlit run streamlit_app.py
```

Opens browser at `http://localhost:8501`

**Features:**
- Chat-based interface
- Real-time streaming responses
- Scrollable conversation history
- Active agent display
- Reset conversation button

### Option 3: API Server (FastAPI)

```bash
python api_server.py
```

Runs on `http://localhost:3000`

**Endpoints:**
- `POST /chat` - Send message and get response
- `POST /reset` - Reset conversation
- `GET /health` - Health check
- `GET /docs` - Interactive API documentation

**Example API call:**

```bash
curl -X POST http://localhost:3000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What time is it?"}'
```

## Multi-Agent System

### Agent Types

1. **General Agent** - Router that decides which specialist to use
2. **Planner Agent** - Handles architecture, planning, and design tasks
3. **Coder Agent** - Handles implementation and debugging
4. **Reviewer Agent** - Handles code review and risk analysis

### How Handoffs Work

1. User sends request to general agent
2. General agent evaluates task and calls `handoff_to_agent()`
3. System updates the active agent and swaps system prompt
4. Specialist agent processes the request with domain expertise
5. Response flows back through conversation history

**Example Flow:**

```
User: "Review this code..."
  ↓
General Agent (sees request is for review)
  ↓
Calls: handoff_to_agent(target_agent="reviewer")
  ↓
System swaps: messages[0] = AGENT_PROMPTS["reviewer"]
  ↓
Reviewer Agent (processes code review)
  ↓
Returns detailed review
```

See `HANDOFF_FLOW.md` for complete technical documentation.

## Tool Capabilities

### Available Tools

- **`get_current_time`** - Returns current UTC time
- **`get_weather`** - Returns weather for specified location
- **`web_search`** - Searches the web for current information
- **`read_file`** - Reads file contents from workspace
- **`write_file`** - Creates or modifies files in workspace
- **`list_files`** - Lists files and directories in workspace
- **`handoff_to_agent`** - Routes to specialist agents

### File I/O Tools

The agent can read, write, and explore files in your workspace:

#### read_file
- Read any supported code file (`.py`, `.ts`, `.js`, `.json`, `.md`, etc.)
- Automatically validates file paths (prevents directory traversal)
- Respects file size limits (default: 100KB)
- Returns file contents and metadata
- **NEW:** Can optionally read from system files (see SYSTEM_FILE_ACCESS.md)

#### write_file
- Create or modify files in workspace
- Automatically creates parent directories
- Validates file extensions for safety
- Returns success confirmation

#### list_files
- Explore directory structure
- Shows file types and sizes
- Helps agent understand project layout

**See `FILE_TOOLS.md` for complete documentation and examples.**

**NEW:** See `SYSTEM_FILE_ACCESS.md` to enable reading files from anywhere on your system!

### Tool Execution

1. Model decides which tools to use based on the task
2. Python executes tools locally
3. Results are added to conversation history
4. Model uses results in next response
5. Agent reports file operations to user

## Guardrails & Safety

### Input Guardrails

- Maximum input length validation
- Repetition detection
- Hateful content blocking (configurable)
- Tool argument validation

### Output Guardrails

- Maximum response length enforcement
- Hateful output filtering
- Tool output sanitization
- Safe fallback messages

Configure via environment variables in `config/settings.py`

## Features

### Web UI Features (Streamlit)

✅ Real-time chat interface  
✅ Scrollable conversation history  
✅ Active agent indicator  
✅ Thinking/processing status  
✅ Web search indicators  
✅ Reset conversation button  
✅ Responsive design  

### API Features

✅ Stateful conversation management  
✅ Health check endpoint  
✅ CORS-enabled for integration  
✅ Full logging  
✅ Error handling with proper HTTP status codes  

### Terminal CLI Features

✅ Interactive input loop  
✅ Debug output for handoffs  
✅ Real-time streaming responses  
✅ Tool execution logging  

## Troubleshooting

### Authentication Issues

**Error:** `401 Unauthorized` or authentication failed

**Solution:**
```bash
# Verify environment variables are set
echo $BEDROCK_API_KEY
echo $OPENAI_BASE_URL

# Reload shell if recently added to .bashrc
source ~/.bashrc

# For Streamlit/API, check they see environment:
python -c "import os; print('BEDROCK_API_KEY:', os.getenv('BEDROCK_API_KEY')[:10] + '...' if os.getenv('BEDROCK_API_KEY') else 'NOT SET')"
```

### Web UI Not Loading Environment Variables

**Issue:** Streamlit/API server can't see `~/.bashrc` variables

**Solution:** Set environment variables in your shell before starting:

```bash
# Load from bashrc
source ~/.bashrc

# Then start the app
streamlit run streamlit_app.py
```

Or add to a `.env` file in project root (see `python-dotenv` support):

```bash
BEDROCK_API_KEY=your-key-here
OPENAI_BASE_URL=https://bedrock-mantle.us-east-1.api.aws/v1
MODEL_ID=openai.gpt-oss-120b
```

### No Tool Calls or Empty Responses

**Issue:** Model isn't calling tools or returning empty responses

**Verify:**
1. Model supports tool calling
2. Request requires external data
3. `MAX_TOOL_ITERATIONS` is sufficient
4. Debug output: set `DEBUG_HANDOFFS=1`

```bash
export DEBUG_HANDOFFS=1
python ai_agents.py
```

### API Server Not Found in Browser UI

**Issue:** Streamlit UI can't reach API server

**Solution:** Ensure both are running in same network context:

```bash
# Terminal 1: Start API
python api_server.py

# Terminal 2: Start UI  
streamlit run streamlit_app.py

# Terminal 3: Start CLI
python ai_agents.py
```

## Configuration Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `BEDROCK_API_KEY` | Required | AWS Bedrock API key |
| `OPENAI_API_KEY` | Optional | Falls back to BEDROCK_API_KEY |
| `OPENAI_BASE_URL` | `https://bedrock-mantle.us-east-1.api.aws/v1` | Bedrock endpoint |
| `MODEL_ID` | `openai.gpt-oss-120b` | Model to use |
| `MAX_USER_INPUT_LENGTH` | 4000 | Input length limit |
| `MAX_TOOL_ITERATIONS` | 8 | Max tool rounds per query |
| `MAX_HANDOFFS_PER_TURN` | 2 | Max agent handoffs |
| `DEBUG_HANDOFFS` | 1 | Show handoff logs |
| `MAX_HISTORY_MESSAGES` | 40 | Conversation memory |
| `WORKSPACE_ROOT` | Current directory | Root for file operations |
| `MAX_FILE_SIZE` | 100 | Max file read size in KB |
| `ALLOWED_READ_EXTENSIONS` | Common code files | File types allowed for reading |
| `ALLOWED_WRITE_EXTENSIONS` | Common code files | File types allowed for writing |

## Examples

### Terminal - Get Weather

```
ask Question.....: what is the weather in dubai?
🔍 Searching...
AI: The weather in Dubai is sunny with a temperature of 42°C (107°F)...
```

### Web UI - Code Review

1. Open `http://localhost:8501`
2. Type: "Review this Python code: `def foo(x): return x + 1`"
3. Watch as:
   - General agent routes to reviewer
   - Reviewer analyzes the code
   - Response displays with formatting

### Terminal - File Operations

```
ask Question.....: Show me the structure of the project
[tool] list_files
AI: The project has the following structure:
- ai_agents.py (main CLI)
- streamlit_app.py (web UI)
- api_server.py (API backend)
- config/settings.py (configuration)
- core/session.py (session management)
...
```

### Terminal - Code Modification

```
ask Question.....: Add error logging to the chat endpoint in api_server.py
[tool] read_file
[tool] write_file
AI: I've added comprehensive error logging to the chat endpoint.
Changes made:
- Added logging import
- Added error handlers for each validation step
- Returns detailed error information in responses
File updated: api_server.py
```

### API - Chat Request

```bash
curl -X POST http://localhost:3000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What time is it?"}' | jq
```

## Notes

- Weather tool uses OpenStreetMap Nominatim + Open-Meteo APIs
- Web search uses DuckDuckGo search integration
- All tool execution happens locally (no model API calls for tools)
- Conversation history is session-based
- Maximum iterations prevent infinite loops

## Additional Documentation

- **FILE_TOOLS_QUICKSTART.md** - Quick start for file I/O tools (5 min)
- **FILE_TOOLS.md** - Complete file I/O tools reference (300+ lines)
- **SYSTEM_FILE_ACCESS.md** - Enable reading files from your system (NEW!)
- **EXTENSION_INTEGRATION.md** - VS Code extension integration guide
- **IMPLEMENTATION_GUIDE.md** - Technical implementation details
- **HANDOFF_FLOW.md** - Deep dive into multi-agent routing system
- **GUARDRILS_INPUT_OUTPUT_FLOW.md** - Safety and guardrails system

## License

MIT

