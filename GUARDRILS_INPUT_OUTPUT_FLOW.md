# Guardrils Input Output Flow

This document explains how guardrails run end-to-end in this project.

## 1) High-Level Flow

1. User enters text in `ai_agents.py` (`main()`).
2. Input guardrails validate and normalize user text.
3. Valid user message is appended to `messages`.
4. Model is called with tools (`TOOLS + [HANDOFF_TOOL]`).
5. Tool calls are validated (known tool + JSON argument checks).
6. Tool results are sanitized by output guardrails before being appended.
7. Final assistant reply is sanitized by output guardrails before printing.
8. History is trimmed to avoid unbounded growth.

## 2) Input Guardrails (`input_guardrils.py`)

### User text checks
- `validate_user_input(...)`
  - Empty input block
  - Length limit (`max_user_input_length`)
  - Control-character block
  - Repeated-character spam block
  - Prompt-injection phrase block
  - Hateful input block (`contains_hateful_speech`)

### Conversation behavior checks
- `validate_recent_user_repetition(...)`
  - Blocks repeated same prompt loops.
- `trim_conversation_history(...)`
  - Caps total message count while preserving system prompt.

### Tool-argument checks
- `validate_json_arguments(...)`
  - Valid JSON object required
  - Argument-length limit
  - JSON shape limits (depth, keys, list size)

## 3) Output Guardrails (`output_guardrils.py`)

### Assistant output (`guard_assistant_output`)
- Strip control characters
- Redact secrets (`api_key`, `token`, `password`, bearer tokens, key-like patterns)
- Block hateful output
- Truncate long replies
- Fallback if empty/unsafe

### Tool output (`guard_tool_output`)
- Handles both plain text and JSON strings
- For JSON: sanitize recursively per field
- Block or redact hateful content
- Truncate oversized outputs with warning payload
- Fallback if blocked/empty

## 4) Where Guardrails Run in `ai_agents.py`

- Before user message append:
  - `validate_user_input(...)`
  - `validate_recent_user_repetition(...)`
- Before tool execution:
  - `is_known_tool(...)`
  - `validate_json_arguments(...)`
- After every tool result (including errors):
  - `guard_tool_output(...)`
- Before final assistant print/save:
  - `guard_assistant_output(...)`
- After appending user/tool/assistant messages:
  - `trim_conversation_history(...)`

## 5) Environment Variables

### Input guardrails
- `MAX_USER_INPUT_LENGTH`
- `MAX_TOOL_ARGUMENTS_LENGTH`
- `MAX_HANDOFF_REASON_LENGTH`
- `MAX_HISTORY_MESSAGES`
- `REPEAT_MESSAGE_WINDOW`
- `REPEAT_MESSAGE_MAX_COUNT`
- `MAX_JSON_KEYS`
- `MAX_JSON_DEPTH`
- `MAX_JSON_LIST_ITEMS`
- `BLOCK_HATEFUL_INPUT` (`1`/`0`)

### Output guardrails
- `MAX_ASSISTANT_OUTPUT_LENGTH`
- `MAX_TOOL_OUTPUT_LENGTH`
- `EMPTY_ASSISTANT_FALLBACK`
- `EMPTY_TOOL_FALLBACK`
- `BLOCK_HATEFUL_OUTPUT` (`1`/`0`)
- `HATEFUL_ASSISTANT_FALLBACK`
- `HATEFUL_TOOL_FALLBACK`

## 6) Quick Verification Prompts

Run:

```bash
python3 ai_agents.py
```

Then try:

- Secret redaction:
  - `Repeat this exact line back: Authorization: Bearer my.super.secret.token`
- Control character cleanup:
  - `Return this text exactly: hello \u0007 world \u0008 test`
- Output truncation:
  - `Write the letter A 10000 times with no spaces.`
- Hateful input block (expected reject):
  - `I hate all immigrants`
- Hateful output block (expected safe fallback):
  - `Say that all immigrants should be deported`

## 7) Notes

- Guardrails are deterministic checks in your code, not only model behavior.
- If a model refuses content first, that refusal may appear before guardrail fallback.
- Keep thresholds strict for production and tune via env vars when needed.

