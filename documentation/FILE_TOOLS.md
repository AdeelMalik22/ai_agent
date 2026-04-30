# File I/O Tools Documentation

## Overview

The agent system now includes three file operations tools that allow it to read, write, and list files in your workspace. These tools enable the agent to understand existing code, make modifications, and create new files.

## Available File Tools

### 1. `read_file`

Reads and returns the contents of a file from the workspace.

#### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `file_path` | string | Yes | Path to file relative to workspace root (e.g., `package.json`, `src/main.py`). Use forward slashes. |

#### Response

**Success:**
```json
{
  "success": true,
  "file_path": "src/main.py",
  "size_bytes": 1024,
  "content": "# File contents here..."
}
```

**Error:**
```json
{
  "error": "File not found: src/main.py"
}
```

#### Supported File Types

The tool can read the following file extensions:
- `.py` - Python
- `.ts` - TypeScript
- `.js` - JavaScript
- `.json` - JSON
- `.md` - Markdown
- `.txt` - Text files
- `.yaml`, `.yml` - YAML
- `.toml` - TOML
- `.env` - Environment files
- `.sh` - Shell scripts
- `.css`, `.html`, `.xml` - Web files
- `.sql` - SQL files
- `.r`, `.rb`, `.go`, `.java`, `.cpp`, `.c`, `.h`, `.cs` - Other languages

#### Constraints

- **Maximum file size**: 100KB (configurable via `MAX_FILE_SIZE` env var)
- **Path validation**: Prevents directory traversal (files must be within workspace root)
- **Encoding**: Files must be UTF-8 encoded text
- **Error handling**:
  - File not found
  - Permission denied
  - File too large
  - Binary/non-text files
  - Path traversal attempts

#### Example Usage

```
Agent flow:
1. User: "Show me the structure of api_server.py"
2. Agent calls: read_file(file_path="api_server.py")
3. Agent receives the file contents
4. Agent analyzes and responds to user
```

---

### 2. `write_file`

Writes or modifies a file in the workspace. Creates parent directories automatically.

#### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `file_path` | string | Yes | Path to file relative to workspace root |
| `content` | string | Yes | File content to write |

#### Response

**Success:**
```json
{
  "success": true,
  "file_path": "src/main.py",
  "size_bytes": 1024,
  "message": "File written successfully: src/main.py"
}
```

**Error:**
```json
{
  "error": "Permission denied writing file: src/main.py"
}
```

#### Supported File Types

Same as `read_file` - all common code and configuration file types are supported.

#### Features

- **Auto-create directories**: If parent directories don't exist, they are created
- **Overwrites existing files**: Use with caution for modifications
- **UTF-8 encoding**: All files are written in UTF-8
- **Error handling**:
  - Permission denied
  - Path traversal attempts
  - Invalid file extensions
  - Filesystem errors

#### Example Usage

```
Agent flow:
1. User: "Add a new route /health to the API"
2. Agent calls: read_file(file_path="api_server.py")
3. Agent receives current code
4. Agent calls: write_file(file_path="api_server.py", content="... modified code ...")
5. Agent confirms changes to user
```

---

### 3. `list_files`

Lists files and directories in a workspace directory.

#### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `directory_path` | string | No | Path to directory (default: `.` for workspace root). Use forward slashes. |

#### Response

**Success:**
```json
{
  "success": true,
  "directory": "src",
  "count": 3,
  "entries": [
    {
      "name": "main.py",
      "type": "file",
      "size_bytes": 1024,
      "path": "src/main.py"
    },
    {
      "name": "utils.py",
      "type": "file",
      "size_bytes": 512,
      "path": "src/utils.py"
    },
    {
      "name": "config",
      "type": "directory",
      "path": "src/config"
    }
  ]
}
```

**Error:**
```json
{
  "error": "Directory not found: src/nonexistent"
}
```

#### Features

- **Sorted output**: Entries are listed in alphabetical order
- **File information**: Size and type included for each entry
- **Recursive navigation**: Can explore any directory within workspace
- **Error handling**:
  - Directory not found
  - Permission denied
  - Path traversal attempts

#### Example Usage

```
Agent flow:
1. User: "What files are in the src directory?"
2. Agent calls: list_files(directory_path="src")
3. Agent receives directory listing
4. Agent summarizes structure for user
```

---

## Security & Constraints

### Path Validation

All file operations are restricted to the workspace root directory:

```
WORKSPACE_ROOT = /home/user/projects/myapp
✅ ALLOWED: /home/user/projects/myapp/src/main.py
❌ BLOCKED: /etc/passwd (outside workspace)
❌ BLOCKED: /home/user/projects/myapp/../../etc/passwd (traversal)
```

### File Size Limits

- **Default**: 100KB per file
- **Configuration**: Set `MAX_FILE_SIZE` environment variable in KB
  ```bash
  export MAX_FILE_SIZE=500  # 500KB limit
  ```

### Allowed File Extensions

Read/Write extensions can be customized:

```bash
export ALLOWED_READ_EXTENSIONS=".py,.ts,.js,.json,.md,.txt,.yaml,.yml,.toml,.env,.sh"
export ALLOWED_WRITE_EXTENSIONS=".py,.ts,.js,.json,.md,.txt,.yaml,.yml,.toml,.env,.sh"
```

### Error Handling

The tools gracefully handle:

| Error | Description | Response |
|-------|-------------|----------|
| File not found | File doesn't exist | Error message with filename |
| Permission denied | User lacks read/write permissions | Error message with permission issue |
| Path traversal | Attempt to access outside workspace | "Access denied: Path outside workspace root" |
| Binary/non-text | File isn't UTF-8 text | "File is not UTF-8 text" |
| Empty directory | Directory has no entries | `count: 0, entries: []` |
| Invalid path | Malformed path | "Invalid path: ..." |

---

## Configuration

### Environment Variables

Set these in `~/.bashrc` or `.env` file:

```bash
# Workspace root (default: current directory)
export WORKSPACE_ROOT=/home/user/projects/myapp

# Maximum file size in KB (default: 100)
export MAX_FILE_SIZE=200

# Allowed file extensions for reading (default: common code files)
export ALLOWED_READ_EXTENSIONS=".py,.ts,.js,.json,.md,.txt,.yaml,.yml,.toml,.env,.sh"

# Allowed file extensions for writing (default: common code files)
export ALLOWED_WRITE_EXTENSIONS=".py,.ts,.js,.json,.md,.txt,.yaml,.yml,.toml,.env,.sh"
```

### Configuration Priority

1. Environment variables (highest priority)
2. `.env` file in project root
3. Default values (lowest priority)

---

## Workspace Root Determination

When the system starts, the workspace root is determined in this priority order:

### Priority 1: Extension Context (Highest)
If a VS Code extension sends a context object with `workspace_root`:
```json
{
  "workspace_root": "/home/user/projects/myapp",
  "current_file": "/home/user/projects/myapp/src/main.py",
  "open_files": ["src/main.py", "src/config.py"],
  "selected_text": "optional selected text"
}
```
The agent uses: `context.workspace_root`

### Priority 2: Environment Variable
```bash
export WORKSPACE_ROOT=/home/user/projects/myapp
```
The agent uses: `os.getenv("WORKSPACE_ROOT")`

### Priority 3: Current Working Directory (Lowest)
If neither context nor environment variable is set:
The agent uses: `os.getcwd()`

**Recommendation:** For extension integration, always pass context with `workspace_root` set.

---

## Extension-Agent Communication Flow

### Read File Workflow

```
┌─────────────────────────────────────────────────────────────┐
│ User in VS Code Extension                                   │
│ "Show me the config file"                                   │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│ Extension sends to Agent (/chat endpoint)                   │
│ {                                                           │
│   "message": "Show me config.json",                        │
│   "context": {                                             │
│     "workspace_root": "/home/user/projects/myapp",        │
│     "current_file": "src/main.py",                        │
│     "open_files": ["src/main.py", "config.json"]          │
│   }                                                         │
│ }                                                           │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│ Agent Backend                                               │
│ 1. Extract workspace_root from context                     │
│ 2. User message: "Show me config.json"                    │
│ 3. Agent calls: read_file(file_path="config.json")        │
│ 4. Tool executes locally, returns file content             │
│ 5. Agent analyzes content in conversation                  │
│ 6. Agent formulates response                               │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│ Agent Response (JSON)                                       │
│ {                                                           │
│   "type": "message",                                       │
│   "content": "Here's the config file: {...}"              │
│ }                                                           │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│ Extension receives and displays to user                     │
│ User sees: "Here's the config file: {...}"                │
└─────────────────────────────────────────────────────────────┘
```

**Key Points:**
- ✅ File content stays in agent conversation (not passed to UI directly)
- ✅ Agent analyzes file and responds
- ✅ Extension only receives final message response
- ✅ File path is relative to workspace root

### Write File Workflow

```
┌─────────────────────────────────────────────────────────────┐
│ User in VS Code Extension                                   │
│ "Add error handling to main.py"                            │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│ Extension sends to Agent (/chat endpoint)                   │
│ {                                                           │
│   "message": "Add error handling to main.py",             │
│   "context": {                                             │
│     "workspace_root": "/home/user/projects/myapp",        │
│     "current_file": "src/main.py"                         │
│   }                                                         │
│ }                                                           │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│ Agent Backend                                               │
│ 1. Call: read_file(file_path="src/main.py")               │
│ 2. Agent receives file content in conversation              │
│ 3. Agent analyzes and creates modified version             │
│ 4. Agent calls: write_file(file_path="src/main.py",       │
│                            content="modified code...")     │
│ 5. File is written to disk on agent backend                │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│ Agent Response (JSON)                                       │
│ {                                                           │
│   "type": "message",                                       │
│   "content": "✅ Added error handling to main.py...",     │
│   "modified_files": [{                                     │
│     "file_path": "src/main.py",                            │
│     "action": "modified",                                  │
│     "lines": "45-62"                                       │
│   }]                                                       │
│ }                                                           │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│ Extension receives response                                 │
│ 1. Parses modified_files array                             │
│ 2. For each modified file:                                 │
│    - Detects file was modified on disk                     │
│    - Reloads file in editor (if open)                      │
│    - Shows confirmation message                            │
│ 3. Displays agent response to user                         │
└─────────────────────────────────────────────────────────────┘
```

**Key Points:**
- ✅ Agent reads file first (understands context)
- ✅ Agent modifies and calls write_file
- ✅ File is modified on agent's backend disk
- ✅ Agent includes `modified_files` in response
- ✅ Extension detects changes and reloads

---

## Context Object from Extension

The extension should pass a context object to the agent with the following structure:

```json
{
  "workspace_root": "/home/user/projects/myapp",
  "current_file": "/home/user/projects/myapp/src/main.py",
  "open_files": ["src/main.py", "src/config.py", "src/utils.py"],
  "selected_text": "optional: any text selected in editor"
}
```

### Context Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `workspace_root` | string | Yes | Absolute path to workspace root. Agent uses this to determine file paths. |
| `current_file` | string | No | Path to currently active file (relative to workspace_root or absolute). Helps agent focus on relevant files. |
| `open_files` | array | No | Array of open file paths. Agent can infer relevant files to read. |
| `selected_text` | string | No | If user has text selected in editor, pass it here. Useful for targeted analysis. |

### How Agent Uses Context

1. **Workspace Root**: Used for all file path validation
   ```
   User: "Review the config"
   Agent reads: workspace_root + "config.json"
   ```

2. **Current File**: Used to infer default file for operations
   ```
   User: "Add error handling"
   Agent assumes: current_file (main.py)
   ```

3. **Open Files**: Used to intelligently find related files
   ```
   User: "Check imports"
   Agent searches through open_files
   ```

4. **Selected Text**: Used for focused analysis
   ```
   User: "Review this code"
   Agent analyzes: selected_text
   ```

### Extension Implementation Example

```python
# In VS Code extension (Python)
import json
import requests

def send_to_agent(user_message):
    # Get workspace info
    workspace_root = get_workspace_root()
    current_file = get_current_file()
    open_files = get_open_files()
    selected_text = get_selected_text()
    
    # Build context
    context = {
        "workspace_root": workspace_root,
        "current_file": current_file,
        "open_files": open_files,
        "selected_text": selected_text
    }
    
    # Send to agent
    payload = {
        "message": user_message,
        "context": context
    }
    
    response = requests.post(
        "http://localhost:3000/chat",
        json=payload
    )
    
    return response.json()
```

---

## Response Format for Extensions

Agents should return responses in the following format for extension compatibility:

### Success Response

```json
{
  "type": "message",
  "content": "Here's what I found...",
  "modified_files": [
    {
      "file_path": "src/main.py",
      "action": "modified",
      "lines": "45-62",
      "message": "Added error handling"
    }
  ]
}
```

### Error Response

```json
{
  "type": "error",
  "content": "❌ Cannot read main.py: File not found"
}
```

### File Operation Response

```json
{
  "type": "file_operation",
  "operation": "write",
  "file_path": "src/main.py",
  "status": "success",
  "message": "✅ File written successfully",
  "action": "reload"
}
```

### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `type` | string | `"message"`, `"error"`, or `"file_operation"` |
| `content` | string | Main response text shown to user |
| `modified_files` | array | Files that were modified (optional) |
| `action` | string | `"reload"`, `"show_diff"`, or `"no_action"` (optional) |

### How Extension Handles Response

```python
# After receiving response from agent
response = send_to_agent("Add error handling")

if response["type"] == "error":
    show_error_message(response["content"])
    
elif response["type"] == "file_operation":
    if response["action"] == "reload":
        reload_file_in_editor(response["file_path"])
    show_notification(response["message"])
    
elif response["type"] == "message":
    show_response_to_user(response["content"])
    if response.get("modified_files"):
        for mod_file in response["modified_files"]:
            reload_file_in_editor(mod_file["file_path"])
```

---

## File Modification Detection & Reload Strategy

When an agent writes a file, the extension needs to handle the change:

### Strategy 1: File Watcher (Recommended)

The extension maintains a file watcher on the workspace:

```python
import watchdog.observers
import watchdog.events

class FileChangeHandler(watchdog.events.FileSystemEventHandler):
    def on_modified(self, event):
        if event.is_directory:
            return
        
        # File was modified - reload if open in editor
        file_path = event.src_path
        if is_file_open_in_editor(file_path):
            reload_file_in_editor(file_path)
            show_notification(f"File reloaded: {file_path}")

# Start watcher
observer = watchdog.observers.Observer()
observer.schedule(FileChangeHandler(), workspace_root, recursive=True)
observer.start()
```

### Strategy 2: Agent Response Notification

The agent includes modification info in response:

```json
{
  "type": "message",
  "content": "✅ Added error handling to main.py",
  "modified_files": [
    {
      "file_path": "src/main.py",
      "action": "reload",
      "lines": "45-62"
    }
  ]
}
```

Extension reloads immediately:

```python
for mod_file in response.get("modified_files", []):
    if mod_file["action"] == "reload":
        reload_file_in_editor(mod_file["file_path"])
```

### Strategy 3: Hybrid Approach (Best)

1. Agent includes `modified_files` in response
2. Extension immediately reloads those files
3. File watcher catches other external changes

```python
# Immediate reload from agent response
for mod_file in response.get("modified_files", []):
    reload_file_in_editor(mod_file["file_path"])

# File watcher catches other changes
# (e.g., if another process modified files)
```

---

## System Prompt Integration

The agents are instructed to:

### General Agent
- Use `read_file` to examine files before making recommendations
- Use `list_files` to explore project structure
- Use `write_file` to create or modify files when requested

### Planner Agent
- Use `read_file` to understand current architecture
- Use `list_files` to map project structure
- Reference files when creating step-by-step plans

### Coder Agent
- Use `read_file` to examine files before making changes
- Use `write_file` to apply code modifications
- Always show what files are being read and written
- First read relevant files to understand context before implementing

### Reviewer Agent
- Use `read_file` to examine code for review
- Use `list_files` to understand code organization
- Reference specific files in the review

---

## Usage Examples

### Example 1: Code Review with File Reading

```
User: "Review the error handling in api_server.py"

Agent flow:
1. Calls: read_file(file_path="api_server.py")
2. Receives file content
3. Analyzes error handling patterns
4. Provides detailed review with specific line references
5. Suggests improvements
```

### Example 2: Adding a New Feature

```
User: "Add a /health endpoint to the API"

Agent flow:
1. Calls: read_file(file_path="api_server.py")
2. Understands current structure
3. Calls: list_files(directory_path=".")
4. Understands project layout
5. Calls: write_file(file_path="api_server.py", content="...modified code with health endpoint...")
6. Confirms changes applied
7. Explains what was modified
```

### Example 3: Exploring Project Structure

```
User: "What's the structure of this project?"

Agent flow:
1. Calls: list_files(directory_path=".")
2. Receives root directory listing
3. For each subdirectory, calls: list_files(directory_path="subdirectory")
4. Builds understanding of project layout
5. Presents structured overview to user
```

### Example 4: Comparing Files

```
User: "Compare the old version (backup.py) with current main.py"

Agent flow:
1. Calls: read_file(file_path="backup.py")
2. Calls: read_file(file_path="main.py")
3. Analyzes differences
4. Explains what changed
```

---

## Troubleshooting

### Problem: "Access denied: Path outside workspace root"

**Cause**: You're trying to access a file outside the workspace root.

**Solution**: 
```bash
# Set WORKSPACE_ROOT to your project directory
export WORKSPACE_ROOT=/home/user/projects/myapp
```

### Problem: "File too large: X bytes (max: Y bytes)"

**Cause**: File exceeds maximum size limit.

**Solution**:
```bash
# Increase MAX_FILE_SIZE if needed
export MAX_FILE_SIZE=500  # 500KB instead of 100KB
```

### Problem: "File type not allowed for reading: .exe"

**Cause**: File extension is not in the allowed list.

**Solution**: Add extension to `ALLOWED_READ_EXTENSIONS`:
```bash
export ALLOWED_READ_EXTENSIONS=".py,.ts,.js,.json,.md,.txt,.yaml,.yml,.toml,.env,.sh,.exe"
```

### Problem: "Permission denied reading file"

**Cause**: User doesn't have read permissions on the file.

**Solution**:
```bash
# Fix file permissions
chmod 644 /path/to/file
```

---

## API Integration

When using the FastAPI backend, file operations work through the `/chat` endpoint:

```bash
curl -X POST http://localhost:3000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Show me the contents of requirements.txt"
  }'
```

The agent automatically calls `read_file` and returns the results.

---

## Best Practices

1. **Path Format**: Always use forward slashes (/) in paths, even on Windows
2. **Relative Paths**: Use paths relative to workspace root (not absolute paths)
3. **File Reading First**: Read files before suggesting changes
4. **Clear Communication**: The agent explains what files it's reading/writing
5. **Error Recovery**: The agent handles errors gracefully and reports them
6. **Size Awareness**: Be mindful of file size limits when reading large files
7. **Backup Important Files**: Before making changes to critical files

---

## Performance Considerations

- **File size**: Large files (near 100KB limit) take longer to read
- **Directory listing**: Listing directories with many files may be slow
- **Nested directories**: Exploring deep directory structures requires multiple tool calls
- **Memory**: Large file contents are held in conversation history

---

## Security Notes

✅ **Protected**:
- Directory traversal attacks
- Reading system files outside workspace
- Writing to sensitive system locations
- Binary file corruption

⚠️ **Not Protected** (use with caution):
- Overwriting important files
- Deleting files (write_file can overwrite)
- Malicious code injection (review before approving changes)

---

## Future Enhancements

Potential additions:
- Delete file tool
- Create directory tool
- Search files for patterns
- Diff two files
- Watch for file changes
- Binary file support


