# Bedrock OpenAI-Compatible Tool Agent

A small interactive Python assistant that uses the OpenAI SDK against an AWS Bedrock OpenAI-compatible endpoint, with local function tool-calling support.

## What this project does

- Uses `OpenAI(...)` with a Bedrock-compatible `base_url`
- Runs an interactive chat loop from `ai_agents.py`
- Exposes tools to the model via function-calling (`tools=TOOLS`)
- Executes tool calls locally through `run_tool(...)` in `tools.py`

## Project files

- `ai_agents.py` - main interactive loop and model call logic
- `system_prompt.py` - system prompt instructions
- `tools.py` - tool schemas and Python tool handlers

## Prerequisites

- Python 3.10+
- AWS Bedrock access and API key/token for your endpoint setup
- Installed Python packages:
  - `openai`
  - `requests`

## Setup

Set environment variables before running:

```bash
export BEDROCK_API_KEY='your-bedrock-api-key'
export OPENAI_BASE_URL='https://bedrock-mantle.us-east-1.api.aws/v1'
export MODEL_ID='openai.gpt-oss-120b'
```

Optional: if you already use `OPENAI_API_KEY`, the app also accepts it.

## Run

```bash
python /home/enigmatix/bed_rock/ai_agents.py
```

In chat:

- Ask normal questions
- Ask tool-friendly questions like:
  - `what time is it in utc?`
  - `what is the weather in dubai in fahrenheit?`
- Type `exit` or `quit` to stop

## How tool-calling works

1. User message is added to conversation history.
2. Model is called with `tools=TOOLS` and `tool_choice="auto"`.
3. If model returns `tool_calls`, Python executes them via `run_tool(...)`.
4. Tool results are added back with `role="tool"`.
5. Model generates final user-facing answer.

## Troubleshooting

- **Authentication errors**: verify `BEDROCK_API_KEY` (or `OPENAI_API_KEY`) is set.
- **Endpoint errors**: verify `OPENAI_BASE_URL` points to your Bedrock OpenAI-compatible endpoint.
- **Model errors**: verify `MODEL_ID` is available in your AWS account/region.
- **No tool calls**: ensure your prompt actually requires external data/action and that the selected model supports tool calling.

## Notes

- Current weather tool uses OpenStreetMap Nominatim + Open-Meteo APIs.
- Keep `User-Agent` configured responsibly for Nominatim usage.

