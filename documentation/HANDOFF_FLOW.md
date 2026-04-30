# Handoff Flow: Complete End-to-End Understanding

This document explains how the multi-agent handoff system works in the Bedrock OpenAI-compatible tool-calling agent, with a complete line-by-line trace through the code.

---

## Architecture Overview

The system has **4 specialist agents**:
- `general` - Router that decides which specialist to handoff to
- `planner` - Handles planning and architecture tasks
- `coder` - Handles implementation and debugging
- `reviewer` - Handles code review and risk analysis

Each agent is a different **system prompt** that guides the model's behavior. The handoff mechanism allows the general agent to route requests to specialists, then control flows back through the same conversation thread.

---

## Key Components

### 1. **system_prompt.py** - Agent Personalities

```python
SYSTEM_PROMPT = """
You are a senior software engineer AI assistant.
...rules and guidelines...
""".strip()

AGENT_PROMPTS = {
    "general": SYSTEM_PROMPT + "\n\nRouting policy:\n..."
    "planner": SYSTEM_PROMPT + "\n\nYou are currently the planning specialist..."
    "coder": SYSTEM_PROMPT + "\n\nYou are currently the coding specialist..."
    "reviewer": SYSTEM_PROMPT + "\n\nYou are currently the review specialist..."
}
```

**What it does:**
- Defines 4 different system prompts (one per agent)
- `general` includes explicit routing rules (planning→planner, code→coder, review→reviewer)
- Each specialist prompt adds role-specific instructions

---

### 2. **tools.py** - Handoff Tool Definition and Execution

#### Part A: The Handoff Tool Schema

```python
HANDOFF_TOOL = {
    "type": "function",
    "function": {
        "name": "handoff_to_agent",
        "description": "Route the current request to a specialist agent. Use this before answering tasks that require planning, coding, or code review expertise.",
        "parameters": {
            "type": "object",
            "properties": {
                "target_agent": {
                    "type": "string",
                    "enum": ["general", "planner", "coder", "reviewer"],
                    "description": "The specialist to hand off to.",
                },
                "reason": {
                    "type": "string",
                    "description": "Short reason for the handoff.",
                },
            },
            "required": ["target_agent"],
        },
    },
}
```

**What it does:**
- Defines the handoff tool that the model can call
- `target_agent` parameter is an enum (restricts to valid agents)
- `reason` is optional (for audit/debugging)
- Exposed to model along with regular tools like `get_weather`, `get_current_time`

#### Part B: The Handoff Executor

```python
def execute_handoff(raw_arguments: str, active_agent: str, handoffs_this_turn: int, max_handoffs: int) -> tuple[str, str, int]:
```

**Input parameters:**
- `raw_arguments` - JSON string containing `{"target_agent": "...", "reason": "..."}`
- `active_agent` - currently active agent (e.g., "general")
- `handoffs_this_turn` - counter tracking how many handoffs happened this turn
- `max_handoffs` - safety limit (default 2) to prevent infinite loops

**Execution flow:**

```python
# Step 1: Parse and validate arguments
args = json.loads(raw_arguments) if raw_arguments else {}
target_agent = args.get("target_agent")
reason = args.get("reason", "")

# Step 2: Validation checks
if not target_agent:
    return error("Missing required argument: target_agent"), active_agent, handoffs_this_turn

if target_agent not in AGENT_PROMPTS:
    return error(f"Unknown target agent: {target_agent}"), active_agent, handoffs_this_turn

if handoffs_this_turn >= max_handoffs:
    return error(f"Handoff limit reached ({max_handoffs})"), active_agent, handoffs_this_turn

if target_agent == active_agent:
    return success("Already on requested agent"), active_agent, handoffs_this_turn

# Step 3: Success - return new agent and increment counter
return success(...), target_agent, handoffs_this_turn + 1
```

**Output:**
- Returns tuple: `(result_json, new_agent, updated_counter)`
- `result_json` contains success/error status
- `new_agent` is the agent to switch to (or current if no change)
- Counter incremented if successful

---

### 3. **ai_agents.py** - Main Orchestration Loop

#### Part A: Initialization (Lines 16-27)

```python
def main() -> None:
    client = build_client()
    model = os.getenv("MODEL_ID", "openai.gpt-oss-120b")
    max_iterations = int(os.getenv("MAX_TOOL_ITERATIONS", "8"))  # max tool rounds per user query
    max_handoffs_per_turn = int(os.getenv("MAX_HANDOFFS_PER_TURN", "2"))  # max handoffs per query
    debug_handoffs = os.getenv("DEBUG_HANDOFFS", "1") == "1"  # debug logging
    active_agent = os.getenv("DEFAULT_AGENT", "general")  # start with general agent
    
    if active_agent not in AGENT_PROMPTS:
        active_agent = "general"  # fallback if invalid
    
    messages = [{"role": "system", "content": AGENT_PROMPTS[active_agent]}]  # init messages with general prompt
```

**Key state:**
- `active_agent` - tracks which agent is currently "active"
- `messages` - conversation history (system, user, assistant, tool results)
- `messages[0]` - always the system prompt (gets swapped when handoff occurs)

#### Part B: User Input Loop (Lines 28-37)

```python
while True:
    print(f"[agent] active={active_agent}")  # debug: show current agent
    user_input = input("ask Question.....: ").strip()
    if not user_input:
        continue
    if user_input.lower() in {"exit", "quit"}:
        print("Bye")
        break
    
    messages.append({"role": "user", "content": user_input})  # add user message to history
    handoffs_this_turn = 0  # reset handoff counter for this new user query
```

**What happens:**
- Outer loop waits for user input
- `handoffs_this_turn` resets to 0 each time user asks a new question
- User message appended to conversation history

#### Part C: Model Invocation (Lines 39-45)

```python
for _ in range(max_iterations):  # max 8 tool rounds
    response = client.chat.completions.create(
        model=model,
        messages=messages,  # full conversation history
        tools=TOOLS + [HANDOFF_TOOL],  # expose both regular tools AND handoff tool
        tool_choice="auto",  # model decides if/when to use tools
    )
```

**What happens:**
- Sends conversation + current system prompt to Bedrock
- Model sees all available tools (weather, time, AND handoff)
- `tool_choice="auto"` means model can choose:
  - Answer directly (no tools), OR
  - Call one/more tools (including handoff)

**Scenario A: General agent + review request**
- User: "review this code: ..."
- System prompt: "if user asks for review... call handoff_to_agent with target_agent='reviewer'"
- Model sees handoff_to_agent tool available
- Model returns: `tool_calls = [handoff_to_agent(target_agent="reviewer")]`

#### Part D: Tool Call Processing (Lines 47-56)

```python
message = response.choices[0].message  # extract assistant's response
tool_calls = message.tool_calls or []  # list of tool calls (may be empty)

if debug_handoffs and tool_calls:
    print(f"[tools] requested={[call.function.name for call in tool_calls]}")

if tool_calls:  # if model requested any tools
    assistant_message = {
        "role": "assistant",
        "content": message.content or "",
        "tool_calls": [tc.model_dump() for tc in tool_calls],
    }
    messages.append(assistant_message)  # add assistant message to history
```

**What happens:**
- Extract tool calls from model response
- If debug enabled, print which tools were called
- Append assistant message (with tool calls info) to history
- This preserves the full tool-call chain for next round

**Example history at this point:**
```
messages = [
    {"role": "system", "content": "You are a senior..."},
    {"role": "user", "content": "review this code..."},
    {"role": "assistant", "content": "...", "tool_calls": [{"function": {"name": "handoff_to_agent", ...}}]}
]
```

#### Part E: Tool Execution Loop (Lines 59-84)

```python
for tool_call in tool_calls:  # iterate each tool call
    if tool_call.function.name == "handoff_to_agent":  # SPECIAL CASE: handoff
        old_agent = active_agent
        result, new_agent, handoffs_this_turn = execute_handoff(
            raw_arguments=tool_call.function.arguments,
            active_agent=active_agent,
            handoffs_this_turn=handoffs_this_turn,
            max_handoffs=max_handoffs_per_turn,
        )
        
        if new_agent != active_agent:  # if handoff successful
            active_agent = new_agent  # UPDATE active agent
            messages[0] = {"role": "system", "content": AGENT_PROMPTS[active_agent]}  # SWAP system prompt!
            if debug_handoffs:
                print(f"[handoff] {old_agent} -> {active_agent}")
        elif debug_handoffs:
            print(f"[handoff] no change ({old_agent})")
    
    else:  # NORMAL TOOL: execute it
        result = run_tool(
            tool_name=tool_call.function.name,
            raw_arguments=tool_call.function.arguments,
        )
    
    # Add tool result to history
    messages.append({
        "role": "tool",
        "tool_call_id": tool_call.id,
        "content": result,
    })
```

**Critical handoff logic:**

1. **Detect handoff call:**
   - Check if `tool_call.function.name == "handoff_to_agent"`
   - This is NOT delegated to `run_tool()` (special in-app handling)

2. **Validate and execute:**
   - Call `execute_handoff()` from tools.py
   - Get back: `(result_json, new_agent, updated_counter)`

3. **Apply handoff (if successful):**
   - **`active_agent = new_agent`** - switch active agent
   - **`messages[0] = AGENT_PROMPTS[active_agent]`** - CRITICAL: replace system prompt
   - This is the key to the handoff: next model call uses new agent's prompt!

4. **Add result to history:**
   - All tools (including handoff) append their result as `role="tool"` message
   - Model sees the result and can plan next action

**Example: After successful handoff**

```
Before handoff:
messages[0] = {"role": "system", "content": AGENT_PROMPTS["general"]}
active_agent = "general"

After handoff to "reviewer":
messages[0] = {"role": "system", "content": AGENT_PROMPTS["reviewer"]}  # CHANGED!
active_agent = "reviewer"

messages = [
    {"role": "system", "content": "You are currently the review specialist..."},
    {"role": "user", "content": "review this code..."},
    {"role": "assistant", "content": "...", "tool_calls": [{"function": {"name": "handoff_to_agent", ...}}]},
    {"role": "tool", "tool_call_id": "...", "content": '{"ok": true, "active_agent": "reviewer"}'}
]
```

#### Part F: Loop Decision (Lines 85-92)

```python
                continue  # if tool_calls, loop back to line 40 with UPDATED messages

        reply = message.content or ""
        print("AI:", reply)
        messages.append({"role": "assistant", "content": reply})
        break  # if NO tool_calls, model gave final answer, exit inner loop

    else:
        print("AI: I hit the tool-iteration limit for this request.")
```

**Control flow:**

- **If tools were called (including handoff):**
  - `continue` - jumps back to `for _ in range(max_iterations):`
  - Next iteration: model sees updated `messages[0]` (new agent prompt)
  - Next iteration: model has context of handoff + tool results
  - Model can now respond as the specialist agent

- **If NO tools were called:**
  - Model gave final text answer in `reply`
  - Print it and break (exit this user's turn)
  - Back to outer `while True:` for next user input

---

## Complete Example: Review Request

### Scenario
User asks: `"Please review this code: data = {"abc":"12"}; y = data.json(); print(y)"`

### Step-by-Step Trace

**State before:**
```
active_agent = "general"
messages[0] = AGENT_PROMPTS["general"]  # contains routing policy
```

**Step 1: User input added (line 36)**
```
messages = [
    {"role": "system", "content": "GENERAL AGENT + routing policy..."},
    {"role": "user", "content": "Please review this code..."}
]
```

**Step 2: Model called (line 40)**
- Model receives system prompt with routing policy
- Model reads: "If user asks for review... call handoff_to_agent with target_agent='reviewer'"
- Model has tools: `[get_weather, get_current_time, handoff_to_agent]`
- Model decides: "This is a review request, I should call handoff_to_agent"
- Model returns: `tool_calls = [{"function": {"name": "handoff_to_agent", "arguments": '{"target_agent": "reviewer", "reason": "Code review requested"}'}}]`

**Step 3: Tool call detected (line 47-56)**
```
tool_calls is not empty, so:
- debug print: [tools] requested=['handoff_to_agent']
- Append to messages:
  {"role": "assistant", "content": "", "tool_calls": [...]}
```

**Step 4: Handoff execution (line 61-72)**
```
tool_call.function.name == "handoff_to_agent" → YES

execute_handoff(
    raw_arguments='{"target_agent": "reviewer"}',
    active_agent="general",
    handoffs_this_turn=0,
    max_handoffs=2
) → returns ("ok", "reviewer", 1)

new_agent = "reviewer" != active_agent="general" → SUCCESS

active_agent = "reviewer"  # SWAP AGENT
messages[0] = AGENT_PROMPTS["reviewer"]  # SWAP SYSTEM PROMPT

debug print: [handoff] general -> reviewer
```

**Step 5: Tool result added (line 81-84)**
```
messages = [
    {"role": "system", "content": "REVIEWER AGENT..."},  # UPDATED!
    {"role": "user", "content": "Please review this code..."},
    {"role": "assistant", "content": "", "tool_calls": [...]},
    {"role": "tool", "tool_call_id": "...", "content": '{"ok": true, "active_agent": "reviewer"}'}
]
```

**Step 6: Loop continues (line 85)**
- `continue` → back to line 40
- Next iteration: model sees reviewer prompt in messages[0]

**Step 7: Second model call (line 40, 2nd iteration)**
- Model now has:
  - System: "You are the review specialist. Focus on bugs, risks..."
  - User: "Please review this code..."
  - Tool history: handoff to reviewer was successful
- Model responds: detailed code review identifying the bug

**Step 8: Final response (line 86-91)**
- Tool calls = empty (model gave direct answer)
- `if tool_calls:` → FALSE
- `reply = "**Code Review**\n\nThe issue is..."`
- Print: `"AI: **Code Review**..."`
- Append: `{"role": "assistant", "content": "..."}`
- `break` → exit inner loop, back to outer while

---

## Key Insights

### 1. **System Prompt Swapping is the Magic**
```python
messages[0] = {"role": "system", "content": AGENT_PROMPTS[active_agent]}
```
By replacing the system prompt in-place, the model gets new instructions without breaking conversation history. The next model call sees the specialist prompt while retaining full context.

### 2. **Handoff is a Local Operation**
- Handoff is **NOT** sent to an external API
- It's handled purely by `execute_handoff()` in Python
- Only result (success/error) is added to messages
- This is cheap (no extra model calls) and keeps control in your app

### 3. **Tool Loop Drives Handoff**
- Each `for _ in range(max_iterations)` is one "round"
- Each round: model sees current state → may call tools
- Handoff is just another tool call
- Next round: model sees new system prompt + tool results
- This creates natural multi-turn agent interactions

### 4. **Counter Prevents Infinite Loops**
```python
if handoffs_this_turn >= max_handoffs:  # default: 2
    return error("limit reached")
```
- Prevents agents from endlessly handing off to each other
- Per-turn limit (resets for each new user question)
- Reasonable default: 2 handoffs max per request

### 5. **Routing Policy Guides Model**
```
"If user asks for planning... call handoff_to_agent with target_agent='planner'"
```
- Without explicit routing, model often answers directly
- Routing policy in system prompt makes handoff "expected behavior"
- Model follows the routing because it's in the instructions

---

## Diagram: Flow Visualization

```
User Input
    ↓
[active_agent="general"]
    ↓
API Call with:
  - messages (history)
  - system="general + routing policy"
  - tools=[weather, time, handoff]
    ↓
Model Response Options:
  ├─ No tools → Final answer → Print & exit
  └─ Tools called → Process tools → Continue
       ├─ Normal tool (weather, time)
       │  └─ run_tool() → External exec → Result
       │
       └─ handoff_to_agent(target="reviewer")
          └─ execute_handoff() → LOCAL ONLY
             ├─ Validate
             ├─ Success? → active_agent="reviewer"
             │            messages[0] = AGENT_PROMPTS["reviewer"]
             └─ Result to messages
    ↓
Next Model Call:
  - messages (updated)
  - system="reviewer prompt"  ← CHANGED!
  - tools=[weather, time, handoff]
    ↓
Model (as Reviewer) → Detailed code review
    ↓
No more tools → Final answer → Print & exit
```

---

## Configuration

Environment variables control handoff behavior:

```bash
export MAX_TOOL_ITERATIONS=8              # max rounds per query (default)
export MAX_HANDOFFS_PER_TURN=2            # max handoffs per query (default)
export DEBUG_HANDOFFS=1                   # print handoff traces (default: on)
export DEFAULT_AGENT=general              # start agent (default)
export MODEL_ID=openai.gpt-oss-120b       # model to use
export OPENAI_BASE_URL=https://...        # Bedrock endpoint
export BEDROCK_API_KEY=...                # Auth
```

To disable debug output:
```bash
export DEBUG_HANDOFFS=0
python ai_agents.py
```

---

## Summary

**The handoff mechanism is elegantly simple:**

1. Model can call `handoff_to_agent(target_agent=X)` tool
2. App intercepts this (doesn't call external API)
3. App updates `active_agent` and swaps `messages[0]` to new prompt
4. Next model call uses specialist prompt with same history
5. Specialist handles the task
6. Conversation continues or ends naturally

**Result:** Multi-agent behavior without complexity, all within a single conversation thread.

