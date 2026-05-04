# New Features & Advanced Setup Guide

This document covers all new features and advanced configurations for the Bedrock OpenAI-Compatible Tool Agent.

---

## Table of Contents

1. [Docker & Docker Compose](#docker--docker-compose)
2. [Agent Performance Metrics](#agent-performance-metrics)
3. [Conversation Export & Import](#conversation-export--import)
4. [Conversation Replay & Debugging](#conversation-replay--debugging)
5. [Configuration Profiles](#configuration-profiles)
6. [Agent Benchmarking Suite](#agent-benchmarking-suite)
7. [Multi-Model Support](#multi-model-support)
8. [Rate Limiting & Cost Tracking](#rate-limiting--cost-tracking)
9. [Interactive Setup Wizard](#interactive-setup-wizard)
10. [Quick Reference](#quick-reference)

---

## Docker & Docker Compose

### Why Docker?
Eliminates "works on my machine" problems. Run the agent in a consistent environment across any system.

### Single Container Setup

#### Build the Docker Image
```bash
docker build -t bedrock-agent:latest .
```

#### Dockerfile Example
Create `Dockerfile` in project root:
```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV WORKSPACE_ROOT=/workspace

# Create workspace volume
VOLUME ["/workspace"]

# Default: CLI mode
CMD ["python", "ai_agents.py"]
```

### Run Modes

#### Terminal/CLI Mode
```bash
docker run -it \
  -e BEDROCK_API_KEY='your-key' \
  -e OPENAI_BASE_URL='https://bedrock-mantle.us-east-1.api.aws/v1' \
  -e MODEL_ID='anthropic.claude-3-sonnet-20240229-v1:0' \
  -v ~/bed_rock:/workspace \
  bedrock-agent:latest
```

#### Web UI Mode
```bash
docker run -it \
  -e BEDROCK_API_KEY='your-key' \
  -p 8501:8501 \
  -v ~/bed_rock:/workspace \
  bedrock-agent:latest \
  streamlit run streamlit_app.py
```

#### API Server Mode
```bash
docker run -it \
  -e BEDROCK_API_KEY='your-key' \
  -p 3000:3000 \
  -v ~/bed_rock:/workspace \
  bedrock-agent:latest \
  python server/api_server.py
```

---

## Docker Compose Full Stack

### What is Docker Compose?
Run all 3 components (CLI, Web UI, API) with one command.

#### docker-compose.yml
```yaml
version: '3.8'

services:
  # Terminal CLI interface
  cli:
    build: .
    container_name: bedrock-cli
    environment:
      BEDROCK_API_KEY: ${BEDROCK_API_KEY}
      OPENAI_BASE_URL: ${OPENAI_BASE_URL}
      MODEL_ID: ${MODEL_ID}
      WORKSPACE_ROOT: /workspace
    volumes:
      - ~/bed_rock:/workspace
    stdin_open: true
    tty: true
    networks:
      - bedrock-network

  # FastAPI Server (backend)
  api:
    build: .
    container_name: bedrock-api
    command: python server/api_server.py
    ports:
      - "3000:3000"
    environment:
      BEDROCK_API_KEY: ${BEDROCK_API_KEY}
      OPENAI_BASE_URL: ${OPENAI_BASE_URL}
      MODEL_ID: ${MODEL_ID}
      WORKSPACE_ROOT: /workspace
    volumes:
      - ~/bed_rock:/workspace
    networks:
      - bedrock-network
    depends_on:
      - cli

  # Streamlit UI (frontend)
  web:
    build: .
    container_name: bedrock-web
    command: streamlit run streamlit_app.py
    ports:
      - "8501:8501"
    environment:
      BEDROCK_API_KEY: ${BEDROCK_API_KEY}
      OPENAI_BASE_URL: ${OPENAI_BASE_URL}
      MODEL_ID: ${MODEL_ID}
      WORKSPACE_ROOT: /workspace
    volumes:
      - ~/bed_rock:/workspace
    networks:
      - bedrock-network
    depends_on:
      - api

networks:
  bedrock-network:
    driver: bridge
```

### Setup & Usage

#### Create .env File
```bash
cat > .env << EOF
BEDROCK_API_KEY=your-key-here
OPENAI_BASE_URL=https://bedrock-mantle.us-east-1.api.aws/v1
MODEL_ID=anthropic.claude-3-sonnet-20240229-v1:0
EOF
```

#### Start Full Stack
```bash
# All services
docker-compose up

# Or in background
docker-compose up -d

# View logs
docker-compose logs -f

# Stop all
docker-compose down
```

#### Start Specific Service
```bash
# Only API
docker-compose up api

# Only Web UI
docker-compose up web

# Only CLI
docker-compose up cli
```

#### Access Services
- **Web UI:** http://localhost:8501
- **API Server:** http://localhost:3000
- **CLI:** Interactive terminal

---

## Agent Performance Metrics

### Overview
Track and analyze how efficiently your agent is operating.

### Enable Metrics
```bash
export ENABLE_METRICS=1
export METRICS_OUTPUT_FILE=metrics.json
```

### Metrics Tracked
- **avg_tools_per_request** – Average tools called per request
- **avg_response_time_ms** – Average response time in milliseconds
- **total_requests** – Total requests processed
- **cache_hit_rate** – Percentage of cached responses
- **handoffs_by_agent** – How many times each agent was used
- **tool_success_rate** – Percentage of successful tool calls
- **avg_tokens_per_request** – Average tokens used

### View Metrics

#### API Endpoint
```bash
curl http://localhost:3000/metrics
```

Response:
```json
{
  "avg_tools_per_request": 2.3,
  "avg_response_time_ms": 1450,
  "total_requests": 142,
  "cache_hit_rate": 0.65,
  "handoffs_by_agent": {
    "coder": 45,
    "reviewer": 38,
    "planner": 12,
    "general": 47
  },
  "tool_success_rate": 0.92,
  "avg_tokens_per_request": 850,
  "peak_response_time_ms": 3200,
  "min_response_time_ms": 280
}
```

#### CLI Report
```bash
python -m metrics report --period today
```

#### Web UI
Metrics appear in Streamlit sidebar automatically when `ENABLE_METRICS=1`

### Performance Optimization
Based on metrics:
- **High tool usage** → Reduce `MAX_TOOL_ITERATIONS`
- **Slow responses** → Use smaller model or optimize tools
- **Low cache hit rate** → Caching not effective, review tool patterns

---

## Conversation Export & Import

### Why Export?
Save conversations for:
- Archiving important interactions
- Sharing with team members
- Auditing AI decisions
- Training datasets
- Legal compliance

### Supported Formats

#### JSON Export (Full Fidelity)
```bash
# Terminal
agent> export json

# Preserves everything:
# - Full message history
# - Tool calls and results
# - Agent routing decisions
# - Timestamps
# - Metadata
```

JSON Structure:
```json
{
  "conversation_id": "conv_2024_04_30_001",
  "created_at": "2024-04-30T14:32:00Z",
  "ended_at": "2024-04-30T14:35:45Z",
  "total_duration_seconds": 225,
  "model": "anthropic.claude-3-sonnet",
  "messages": [
    {
      "role": "user",
      "content": "Review my code",
      "timestamp": "2024-04-30T14:32:00Z"
    },
    {
      "role": "assistant",
      "content": "I'll review your code...",
      "agent": "coder",
      "tools_used": ["read_file"],
      "timestamp": "2024-04-30T14:32:15Z"
    }
  ],
  "metrics": {
    "total_tokens": 2450,
    "tool_calls": 3,
    "handoffs": 1
  }
}
```

#### Markdown Export (Human-Readable)
```bash
# Terminal
agent> export markdown

# Output:
# conversation_2024_04_30_001.md
```

Markdown Structure:
```markdown
# Conversation: Code Review
**Date:** April 30, 2024  
**Duration:** 3 minutes 45 seconds  
**Model:** Claude 3 Sonnet  

## Messages

### User (14:32:00)
Review my code

### Assistant - Coder Agent (14:32:15)
I'll review your code by examining the files...

**Tools Used:** read_file  
**Agent:** coder  

### Tool Result: read_file
```python
def hello():
    print("Hello World")
```

### Assistant (14:32:45)
Your code is clean and simple...
```

#### PDF Export (Formatted Report)
```bash
# Terminal
agent> export pdf

# Output:
# conversation_2024_04_30_001.pdf
# Includes: title, metadata, formatted messages, charts
```

### API Export
```python
import requests

response = requests.post(
    "http://localhost:3000/export",
    json={
        "format": "markdown",  # or "json", "pdf"
        "include_metadata": True,
        "conversation_id": "conv_2024_001"
    }
)

# Get exported file
with open("conversation.md", "w") as f:
    f.write(response.text)
```

### Auto-Export
```bash
# Save all conversations automatically
export AUTO_EXPORT=true
export EXPORT_FORMAT=markdown
export EXPORT_DIR=~/conversations
```

---

## Conversation Replay & Debugging

### Why Replay?
Debugging agent behavior:
- See exactly what the agent did
- Understand reasoning at each step
- Modify and re-run from any point
- Test agent changes on past conversations

### Start Replay
```bash
# List conversations
python -m replay list

# Replay specific conversation
python -m replay conv_2024_04_30_001
```

### Interactive Replay Commands
```
[Step 1/24] General Agent routing request...

agent> help
Commands:
  next         - Go to next step
  prev         - Go to previous step
  jump N       - Jump to step N
  show         - Show current state
  tools        - Show tools called
  edit         - Edit and re-run from here
  save         - Save new conversation
  quit         - Exit replay

agent> show
Step 1/24: User Input
Message: "Review my code"
Agent: general
Action: Route to specialist

agent> jump 5
[Step 5/24] Coder Agent reading files...

agent> tools
Tools called so far:
  1. list_files(directory_path=".")
  2. read_file(file_path="main.py")
  3. read_file(file_path="utils.py")

agent> edit
# Modify the system prompt for testing
# Current prompt: ...
# Edit? (y/n): y
[Editor opens]
```

### Export Replay Report
```bash
python -m replay conv_2024_04_30_001 --export report.md
```

Report shows:
- Step-by-step execution
- Tool outputs at each step
- Agent reasoning
- Timing information
- Performance metrics

---

## Configuration Profiles

### What are Profiles?
Pre-configured settings for different use cases.

### Available Profiles

#### profiles/development.env
Optimized for active development:
```bash
# Aggressive tool usage, detailed debugging
MAX_TOOL_ITERATIONS=12
DEBUG=true
LOG_LEVEL=DEBUG
VERBOSE_OUTPUT=true
ENABLE_METRICS=true
CACHE_RESPONSES=false
SAFETY_MODE=permissive
```

#### profiles/production.env
Safe, efficient, monitored:
```bash
# Conservative, high safety, monitoring
MAX_TOOL_ITERATIONS=5
DEBUG=false
LOG_LEVEL=WARNING
VERBOSE_OUTPUT=false
ENABLE_METRICS=true
CACHE_RESPONSES=true
SAFETY_MODE=strict
RATE_LIMIT_PER_MINUTE=10
```

#### profiles/fast.env
Maximum speed, minimal overhead:
```bash
# Speed-optimized, fewer tools
MAX_TOOL_ITERATIONS=3
CACHE_RESPONSES=true
STREAM_RESPONSE=true
PARALLEL_TOOLS=true
SAFETY_MODE=standard
```

#### profiles/thorough.env
Maximum accuracy, detailed analysis:
```bash
# Uses all tools, deep analysis
MAX_TOOL_ITERATIONS=15
ENABLE_METRICS=true
DETAILED_LOGGING=true
CACHE_RESPONSES=false
SAFETY_MODE=strict
RESEARCH_MODE=true
```

### Use a Profile

#### Via Environment
```bash
source profiles/production.env
python ai_agents.py
```

#### Via Command Line
```bash
python ai_agents.py --profile fast
```

#### Via Docker
```bash
docker run \
  --env-file profiles/production.env \
  bedrock-agent:latest
```

### Create Custom Profile
```bash
# Copy template
cp profiles/production.env profiles/custom.env

# Edit with your settings
nano profiles/custom.env

# Use it
source profiles/custom.env
python ai_agents.py
```

---

## Agent Benchmarking Suite

### What is Benchmarking?
Measure agent quality, speed, and consistency.

### Available Benchmarks

#### Benchmark: Code Review
```bash
python -m benchmark \
  --task code_review \
  --iterations 10 \
  --profile production
```

Measures:
- Accuracy (catches bugs?)
- Speed (avg response time)
- Tool efficiency (tools used / quality)
- Consistency (similar results each run)

#### Benchmark: Document Analysis
```bash
python -m benchmark \
  --task document_analysis \
  --iterations 5 \
  --samples 20
```

#### Benchmark: Web Search
```bash
python -m benchmark \
  --task web_search \
  --queries 15
```

#### Benchmark: File Operations
```bash
python -m benchmark \
  --task file_operations \
  --file_sizes small,medium,large
```

### View Results
```bash
# Latest report
python -m benchmark report --latest

# Specific date
python -m benchmark report --date 2024-04-30

# Compare profiles
python -m benchmark compare \
  --profile1 production \
  --profile2 fast
```

### Report Output
```
╔════════════════════════════════════════╗
║        Benchmark Results               ║
║        Code Review (10 iterations)     ║
╠════════════════════════════════════════╣
║ Accuracy:            92.5%             ║
║ Avg Response Time:   1.45s             ║
║ Tool Efficiency:     0.87 (good)      ║
║ Consistency:         94% similar       ║
║ Bugs Caught:         9/10             ║
║ False Positives:     1                ║
║ Avg Tokens Used:     850              ║
╠════════════════════════════════════════╣
║ Profile: production                    ║
║ Model: Claude 3 Sonnet                 ║
║ Date: 2024-04-30                       ║
╚════════════════════════════════════════╝
```

### Regression Testing
```bash
# Compare against baseline
python -m benchmark \
  --task code_review \
  --compare baseline
```

Alerts if:
- Performance dropped > 5%
- Accuracy dropped > 2%
- Response time increased > 10%

---

## Multi-Model Support

### Supported Providers

#### AWS Bedrock (Default)
```bash
export MODEL_PROVIDER=bedrock
export BEDROCK_API_KEY=your-key
export MODEL_ID=anthropic.claude-3-sonnet-20240229-v1:0
```

#### OpenAI API
```bash
export MODEL_PROVIDER=openai
export OPENAI_API_KEY=sk-...
export MODEL_ID=gpt-4-turbo
```

#### Anthropic Claude (Direct)
```bash
export MODEL_PROVIDER=anthropic
export ANTHROPIC_API_KEY=sk-ant-...
export MODEL_ID=claude-3-opus-20240229
```

#### Local Ollama
```bash
export MODEL_PROVIDER=ollama
export OLLAMA_BASE_URL=http://localhost:11434
export MODEL_ID=llama2
```

### Switch Providers

#### Via Environment
```bash
export MODEL_PROVIDER=openai
python ai_agents.py
```

#### Via Command Line
```bash
python ai_agents.py --provider openai --model gpt-4
```

#### Via Docker
```bash
docker run \
  -e MODEL_PROVIDER=anthropic \
  -e ANTHROPIC_API_KEY=your-key \
  bedrock-agent:latest
```

### Provider-Specific Configuration

#### OpenAI
```bash
export MODEL_PROVIDER=openai
export OPENAI_API_KEY=sk-...
export OPENAI_ORG_ID=optional-org-id
export OPENAI_REQUEST_TIMEOUT=30
export OPENAI_MAX_RETRIES=3
```

#### Anthropic
```bash
export MODEL_PROVIDER=anthropic
export ANTHROPIC_API_KEY=sk-ant-...
export ANTHROPIC_TIMEOUT=60
```

#### Ollama
```bash
export MODEL_PROVIDER=ollama
export OLLAMA_BASE_URL=http://localhost:11434
export OLLAMA_MODEL=llama2:13b
```

### Model Comparison
```bash
python -m compare_models \
  --models gpt-4,claude-3-opus,llama2 \
  --task code_review \
  --iterations 5
```

Output:
```
╔════════════════════════════════════════════════════╗
║            Model Comparison Results                ║
╠═══════════════════╦═════════╦════════╦═════════════╣
║ Model             ║ Speed   ║ Quality║ Cost/1K     ║
╠═══════════════════╬═════════╬════════╬═════════════╣
║ GPT-4 Turbo       ║ 1.2s    ║ 95%    ║ $0.030      ║
║ Claude 3 Opus     ║ 0.9s    ║ 94%    ║ $0.015      ║
║ Llama 2 (Ollama)  ║ 2.1s    ║ 81%    ║ $0.000      ║
╚═══════════════════╩═════════╩════════╩═════════════╝
```

---

## Rate Limiting & Cost Tracking

### Overview
Monitor API usage and control spending.

### Enable Cost Tracking
```bash
export ENABLE_COST_TRACKING=1
export MONTHLY_BUDGET_USD=100
export COST_ALERT_THRESHOLD=0.8  # Alert at 80% of budget
```

### Track Costs

#### CLI Report
```bash
python -m usage_report --month current
```

Output:
```
╔════════════════════════════════════╗
│   Cost Report (April 2024)         │
├════════════════════════════════════╤
│ Total Requests: 1,245              │
│ API Calls: 3,120                   │
│ Tokens Used: 2,450,000             │
│ Estimated Cost: $12.45             │
│ Budget: $100                       │
│ Budget Used: 12.5%                 │
│ Projected (Full Month): $38.95     │
├────────────────────────────────────┤
│ Breakdown by Provider:             │
│ • Bedrock: $8.50 (68%)            │
│ • OpenAI: $3.95 (32%)             │
│                                    │
│ Top Tools by Cost:                 │
│ • read_file: $4.20 (34%)          │
│ • web_search: $3.50 (28%)         │
│ • model_calls: $4.75 (38%)        │
╚════════════════════════════════════╝
```

#### API Endpoint
```bash
curl http://localhost:3000/cost-report
```

### Rate Limiting

#### Per-Minute Limit
```bash
export RATE_LIMIT_PER_MINUTE=10
```

Behavior when exceeded:
- Request queued with 429 response
- Automatic retry with backoff
- User notified in UI

#### Per-User Limits
```bash
# In API server configuration
RATE_LIMITS = {
    "default": 10,      # 10 req/min
    "premium": 50,      # 50 req/min
    "admin": 1000       # 1000 req/min
}
```

#### Budget Enforcement
```bash
export ENABLE_COST_TRACKING=1
export MONTHLY_BUDGET_USD=100

# When budget exceeded:
# - All new requests rejected with 403
# - Error: "Monthly budget limit exceeded"
# - Admin must reset budget
```

### Cost Optimization Tips
```bash
# 1. Use smaller models for simple tasks
export MODEL_ID=gpt-3.5-turbo  # Cheaper than gpt-4

# 2. Enable response caching
export CACHE_RESPONSES=true

# 3. Reduce tool iterations
export MAX_TOOL_ITERATIONS=5

# 4. Batch process requests
# Process 10 documents in 1 session vs 10 sessions

# 5. Monitor usage regularly
python -m usage_report --period weekly
```

---

## Interactive Setup Wizard

### Why a Wizard?
Automates complex configuration for non-technical users.

### Start Wizard
```bash
python -m setup
```

### Wizard Steps

#### Step 1: Check Environment
```
Checking Python version...
✓ Python 3.10.8 (meets requirement: 3.10+)

Checking system resources...
✓ RAM: 8.2 GB available (need 4GB)
✓ Disk: 150GB free (need 500MB)
✓ Internet: Connected
```

#### Step 2: Verify AWS Credentials
```
Checking AWS credentials...
✓ AWS Access Key ID found
✓ AWS Secret Access Key found
✓ AWS Region: us-east-1

Test Bedrock connection? (y/n): y
✓ Connected to AWS Bedrock
✓ Available models:
  - anthropic.claude-3-sonnet-20240229-v1:0
  - anthropic.claude-3-opus-20240229-v1:0
  - amazon.titan-text-express-v1:0
```

#### Step 3: Select Model
```
Which model do you prefer?

1) Claude 3 Sonnet (Fast, good quality) [RECOMMENDED]
2) Claude 3 Opus (Slower, better quality)
3) Titan Text (Cheaper, okay quality)

Enter choice (1-3): 1
✓ Selected: anthropic.claude-3-sonnet-20240229-v1:0
```

#### Step 4: Configure Interface
```
Which interfaces to enable?

[x] Terminal CLI (recommended)
[x] Web UI (Streamlit)
[x] API Server (FastAPI)

Continue? (y/n): y
```

#### Step 5: Set Workspace
```
Workspace directory? (default: /home/username/bed_rock)
> /home/username/bed_rock
✓ Workspace set: /home/username/bed_rock
```

#### Step 6: Configure Settings
```
Max file size to read? (default: 100 KB)
> 200

Allow system file access? (default: no)
> no

Max tool iterations per request? (default: 8)
> 8
```

#### Step 7: Test Configuration
```
Running test query...
✓ Initialized session
✓ Connected to model
✓ Tested tool execution

Test message: "Hello, what is 2+2?"
✓ Got response: "2+2 equals 4"

All tests passed! ✓
```

#### Step 8: Generate Configuration
```
Generating configuration files...
✓ Created: .env
✓ Created: profiles/custom.env
✓ Created: config/local_settings.py

Configuration complete!
```

#### Step 9: Save & Start
```
Configuration saved to .env

Ready to start?

1) Start Terminal CLI
2) Start Web UI
3) Start API Server
4) Exit wizard

Enter choice (1-4): 1
Starting Terminal CLI...
```

### Wizard Output
Generated files:
- `.env` – Environment variables
- `config/local_settings.py` – Local configuration
- `profiles/wizard-config.env` – Reusable profile

---

## Quick Reference

### Common Commands

#### Start Agent (Different Modes)
```bash
# Terminal CLI
python ai_agents.py

# Web UI
streamlit run streamlit_app.py

# API Server
python server/api_server.py
```

#### Docker
```bash
# Build image
docker build -t bedrock-agent .

# Run with Docker
docker run -e BEDROCK_API_KEY=key bedrock-agent

# Run full stack
docker-compose up
```

#### Export Conversation
```bash
# Terminal
agent> export markdown

# CLI
python -m export --format json --output conv.json
```

#### Replay Conversation
```bash
python -m replay conv_id
```

#### View Metrics
```bash
curl http://localhost:3000/metrics
```

#### Run Benchmark
```bash
python -m benchmark --task code_review --iterations 10
```

#### View Cost Report
```bash
python -m usage_report --month current
```

#### Use Different Profile
```bash
source profiles/production.env
python ai_agents.py
```

#### Compare Models
```bash
python -m compare_models --models gpt-4,claude-3-opus
```

### Environment Variables Cheat Sheet

```bash
# Required
BEDROCK_API_KEY=your-key
MODEL_ID=anthropic.claude-3-sonnet-20240229-v1:0

# Recommended
WORKSPACE_ROOT=/home/username/bed_rock
OPENAI_BASE_URL=https://bedrock-mantle.us-east-1.api.aws/v1

# Optional
MAX_TOOL_ITERATIONS=8
ENABLE_METRICS=1
ENABLE_COST_TRACKING=1
MONTHLY_BUDGET_USD=100
CACHE_RESPONSES=true
MODEL_PROVIDER=bedrock
RATE_LIMIT_PER_MINUTE=10
AUTO_EXPORT=true
EXPORT_FORMAT=markdown
```

### Troubleshooting New Features

#### Metrics Not Showing
```bash
export ENABLE_METRICS=1
# Restart agent
python ai_agents.py
```

#### Docker Port Already in Use
```bash
# Use different port
docker run -p 8502:8501 bedrock-agent:latest streamlit run streamlit_app.py
```

#### Cost Tracking Too High
```bash
# Reduce tool usage
export MAX_TOOL_ITERATIONS=3

# Enable caching
export CACHE_RESPONSES=true

# Use cheaper model
export MODEL_ID=gpt-3.5-turbo
```

#### Replay Not Working
```bash
# Ensure conversations are saved
export SAVE_CONVERSATIONS=true
# Check conversation log
ls -la ~/.bedrock/conversations/
```

---

## Summary

| Feature | Benefit | Command |
|---------|---------|---------|
| Docker | Portable, reproducible | `docker build -t bedrock-agent .` |
| Metrics | Optimization insights | `curl :3000/metrics` |
| Export | Data portability | `agent> export markdown` |
| Replay | Debugging | `python -m replay conv_id` |
| Profiles | Quick config switching | `source profiles/production.env` |
| Benchmark | Quality assurance | `python -m benchmark --task code_review` |
| Multi-Model | Provider flexibility | `export MODEL_PROVIDER=openai` |
| Cost Tracking | Budget control | `python -m usage_report` |
| Setup Wizard | Easy onboarding | `python -m setup` |

---

## Getting Help

- 📖 Check this documentation
- 🐛 Report issues with detailed info
- 💬 Ask in discussions
- 📧 Contact support

---

*Last Updated: April 30, 2024*
*For Bedrock OpenAI-Compatible Tool Agent*

