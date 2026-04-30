# VS Code Extension Integration Guide

## Overview

This guide explains how to integrate a VS Code extension with the Bedrock OpenAI-compatible Tool Agent to enable file I/O operations directly from the editor.

## Architecture

```
┌──────────────────────────┐
│  VS Code Extension       │
│  (TypeScript/JavaScript) │
└────────────┬─────────────┘
             │ HTTP POST /chat
             │
┌────────────▼─────────────┐
│  FastAPI Backend         │
│  (Python)                │
│  - api_server.py         │
│  - Port: 3000            │
└────────────┬─────────────┘
             │ File operations
             │
┌────────────▼─────────────┐
│  Agent Backend           │
│  - tools.py              │
│  - File I/O tools        │
└──────────────────────────┘
```

## API Endpoint: `/chat`

### Request Format

```json
{
  "message": "Show me the main.py file",
  "context": {
    "workspace_root": "/home/user/projects/myapp",
    "current_file": "/home/user/projects/myapp/src/main.py",
    "open_files": ["src/main.py", "src/config.py"],
    "selected_text": "optional selected code"
  }
}
```

### Response Format

```json
{
  "type": "message",
  "content": "Here's the main.py file content...",
  "agent": "general",
  "modified_files": [
    {
      "file_path": "src/main.py",
      "action": "modified",
      "lines": "45-62"
    }
  ]
}
```

## Implementation Steps

### Step 1: Setup Agent Backend

Ensure the FastAPI server is running:

```bash
# Terminal 1: Start the API server
python api_server.py
# Server runs on http://localhost:3000
```

### Step 2: Create Extension

Create a VS Code extension that sends requests to the agent:

```typescript
// extension.ts (TypeScript example)

import * as vscode from 'vscode';
import axios from 'axios';

const AGENT_API = 'http://localhost:3000/chat';

export function activate(context: vscode.ExtensionContext) {
    const disposable = vscode.commands.registerCommand(
        'extension.askAgent',
        async () => {
            const message = await vscode.window.showInputBox({
                prompt: 'Ask the agent...'
            });

            if (message) {
                const response = await sendToAgent(message);
                vscode.window.showInformationMessage(response);
            }
        }
    );

    context.subscriptions.push(disposable);
}

async function sendToAgent(message: string) {
    const workspaceRoot = vscode.workspace.rootPath || '';
    const currentFile = vscode.window.activeTextEditor?.document.uri.fsPath || '';
    const openFiles = vscode.workspace.textDocuments.map(doc => 
        doc.uri.fsPath.replace(workspaceRoot + '/', '')
    );

    const payload = {
        message: message,
        context: {
            workspace_root: workspaceRoot,
            current_file: currentFile,
            open_files: openFiles,
            selected_text: vscode.window.activeTextEditor?.document.getText(
                vscode.window.activeTextEditor.selection
            ) || ''
        }
    };

    const response = await axios.post(AGENT_API, payload);
    return response.data.content;
}
```

### Step 3: Handle File Modifications

When the agent modifies files, the extension should detect and reload:

```typescript
// Monitor file changes
async function setupFileWatcher() {
    const watcher = vscode.workspace.createFileSystemWatcher('**/*');

    watcher.onDidChange(async (uri) => {
        // File was modified - reload if open
        const editor = vscode.window.visibleTextEditors.find(
            e => e.document.uri.fsPath === uri.fsPath
        );
        
        if (editor) {
            // Reload the document from disk
            const document = await vscode.workspace.openTextDocument(uri);
            await vscode.window.showTextDocument(document);
        }
    });

    return watcher;
}
```

### Step 4: Display Agent Responses

Format and display agent responses in the extension:

```typescript
async function showAgentResponse(response: any) {
    const panel = vscode.window.createWebviewPanel(
        'agentResponse',
        'Agent Response',
        vscode.ViewColumn.Two
    );

    panel.webview.html = `
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body { font-family: sans-serif; padding: 20px; }
                .response { background: #f5f5f5; padding: 10px; border-radius: 5px; }
                .modified { color: green; }
                .error { color: red; }
            </style>
        </head>
        <body>
            <div class="response">
                ${response.content}
            </div>
            ${response.modified_files ? `
                <div class="modified">
                    <p><strong>Modified files:</strong></p>
                    <ul>
                        ${response.modified_files.map((f: any) => 
                            `<li>${f.file_path} (lines ${f.lines})</li>`
                        ).join('')}
                    </ul>
                </div>
            ` : ''}
        </body>
        </html>
    `;
}
```

## File I/O Workflows

### Read File

**User Action:**
```
1. User selects "Show me config.json" from extension menu
2. Extension sends to agent with context
3. Agent calls read_file(file_path="config.json")
4. Agent analyzes and responds
5. Extension displays response
```

**Extension Code:**
```typescript
const response = await sendToAgent("Show me the config file");
showAgentResponse(response);
```

### Write File

**User Action:**
```
1. User asks: "Add error handling to main.py"
2. Extension sends to agent
3. Agent reads main.py (via read_file)
4. Agent modifies and calls write_file
5. File modified on disk
6. Extension detects change and reloads
```

**Extension Handling:**
```typescript
const response = await sendToAgent("Add error handling to main.py");

// If files were modified, reload them
if (response.modified_files) {
    for (const file of response.modified_files) {
        const fileUri = vscode.Uri.file(file.file_path);
        const document = await vscode.workspace.openTextDocument(fileUri);
        await vscode.window.showTextDocument(document);
    }
}

showAgentResponse(response);
```

### List Files

**User Action:**
```
1. User asks: "Show me the project structure"
2. Agent calls list_files(directory_path=".")
3. Agent formats and responds with structure
4. Extension displays in a panel
```

## Context Object Details

### What to Send

```json
{
  "workspace_root": "/absolute/path/to/workspace",
  "current_file": "/absolute/path/to/current/file.py",
  "open_files": [
    "src/main.py",
    "src/config.py",
    "src/utils.py"
  ],
  "selected_text": "code snippet if selected"
}
```

### Why Each Field Matters

| Field | Purpose | Example |
|-------|---------|---------|
| `workspace_root` | Root for all file operations | Validates paths, prevents traversal |
| `current_file` | Default file for operations | "Add logging" → assumes current file |
| `open_files` | Related files to consider | For "find all imports" type tasks |
| `selected_text` | Focused analysis scope | "Review this" → analyzes selection |

### How to Collect

```typescript
function getContext(): any {
    return {
        workspace_root: vscode.workspace.rootPath || '',
        current_file: vscode.window.activeTextEditor?.document.uri.fsPath || '',
        open_files: vscode.workspace.textDocuments.map(doc => {
            // Return relative paths
            const root = vscode.workspace.rootPath || '';
            return doc.uri.fsPath.replace(root + '/', '');
        }),
        selected_text: getSelectedText()
    };
}

function getSelectedText(): string {
    const editor = vscode.window.activeTextEditor;
    if (editor && !editor.selection.isEmpty) {
        const text = editor.document.getText(editor.selection);
        return text;
    }
    return '';
}
```

## Error Handling

Handle different response types from the agent:

```typescript
async function handleAgentResponse(response: any) {
    switch (response.type) {
        case 'error':
            vscode.window.showErrorMessage(response.content);
            break;
            
        case 'file_operation':
            if (response.action === 'reload') {
                reloadFile(response.file_path);
            }
            vscode.window.showInformationMessage(response.message);
            break;
            
        case 'message':
            showAgentResponse(response);
            // Handle modified files if present
            if (response.modified_files) {
                for (const file of response.modified_files) {
                    reloadFile(file.file_path);
                }
            }
            break;
    }
}

function reloadFile(filePath: string) {
    const fileUri = vscode.Uri.file(filePath);
    vscode.workspace.openTextDocument(fileUri).then(doc => {
        vscode.window.showTextDocument(doc);
    });
}
```

## Configuration

### Agent Server URL

Make configurable via extension settings:

```json
{
  "bedrockAgent.serverUrl": "http://localhost:3000",
  "bedrockAgent.autoReload": true,
  "bedrockAgent.showNotifications": true
}
```

```typescript
function getAgentUrl(): string {
    const config = vscode.workspace.getConfiguration('bedrockAgent');
    return config.get('serverUrl', 'http://localhost:3000');
}
```

### Workspace Root Detection

Priority order:

1. Use `vscode.workspace.rootPath` (highest priority)
2. Allow user to configure in settings
3. Use first opened folder (lowest priority)

```typescript
function getWorkspaceRoot(): string {
    // Priority 1: Current workspace
    if (vscode.workspace.rootPath) {
        return vscode.workspace.rootPath;
    }
    
    // Priority 2: Configuration
    const config = vscode.workspace.getConfiguration('bedrockAgent');
    const configured = config.get('workspaceRoot');
    if (configured) {
        return configured as string;
    }
    
    // Priority 3: First folder
    if (vscode.workspace.workspaceFolders && 
        vscode.workspace.workspaceFolders.length > 0) {
        return vscode.workspace.workspaceFolders[0].uri.fsPath;
    }
    
    return '';
}
```

## Testing

### Test the Extension

1. **Start agent backend:**
   ```bash
   python api_server.py
   ```

2. **Test API endpoint:**
   ```bash
   curl -X POST http://localhost:3000/chat \
     -H "Content-Type: application/json" \
     -d '{
       "message": "What files are in src?",
       "context": {
         "workspace_root": "/home/user/projects/myapp"
       }
     }'
   ```

3. **Test from extension:**
   - Open VS Code with extension
   - Run command "Ask Agent"
   - Type a question
   - Verify response displays

### Test File Operations

1. **Test read:**
   ```
   Message: "Show me the README.md file"
   Expected: File content displayed
   ```

2. **Test write:**
   ```
   Message: "Add a health endpoint to api_server.py"
   Expected: 
   - File modified on disk
   - Extension reloads file
   - Confirmation message shown
   ```

3. **Test list:**
   ```
   Message: "Show me the project structure"
   Expected: Directory listing with files and folders
   ```

## Troubleshooting

### Agent Not Responding

**Problem:** Extension can't reach agent backend

**Solution:**
```bash
# Check if server is running
curl http://localhost:3000/health

# Start server if not running
python api_server.py

# Check firewall if on different machines
# Ensure port 3000 is open
```

### Files Not Reloading

**Problem:** Extension doesn't reload modified files

**Solution:**
1. Ensure file watcher is active
2. Check that `modified_files` is in response
3. Verify file path is correct

```typescript
// Add logging
console.log('Modified files:', response.modified_files);
```

### Paths Not Resolving

**Problem:** Agent can't find files

**Solution:**
1. Verify `workspace_root` is set correctly
2. Check file paths are relative to workspace_root
3. Use forward slashes (/) in paths

```typescript
// Debug context
console.log('Context:', getContext());
```

### Workspace Root Confusion

**Problem:** Agent says "Path outside workspace root"

**Solution:**
```typescript
// Ensure absolute path for workspace_root
const root = vscode.workspace.rootPath;
if (root) {
    console.log('Workspace root:', root); // Should be /home/user/projects/myapp
}
```

## Performance Tips

1. **Cache file content** - Don't request same file multiple times
2. **Batch operations** - Combine multiple requests when possible
3. **Limit context size** - Don't send entire workspace file list
4. **Debounce requests** - Wait for user to finish typing before sending

```typescript
let requestTimeout: any;

function onUserInput(message: string) {
    clearTimeout(requestTimeout);
    requestTimeout = setTimeout(() => {
        sendToAgent(message);
    }, 500); // Wait 500ms after typing stops
}
```

## Security Considerations

1. **Validate responses** - Check agent responses before acting on them
2. **Confirm file modifications** - Ask user before auto-reloading important files
3. **Workspace isolation** - Only access files within workspace_root
4. **Limit file types** - Restrict to safe extensions (code files only)

```typescript
const SAFE_EXTENSIONS = ['.py', '.ts', '.js', '.json', '.md', '.txt'];

function isSafeFile(filePath: string): boolean {
    const ext = filePath.substring(filePath.lastIndexOf('.'));
    return SAFE_EXTENSIONS.includes(ext.toLowerCase());
}
```

## Example: Complete Extension

See the companion extension repository for a complete working example.

## Support

- **Agent Issues:** Check FILE_TOOLS.md in main repository
- **API Issues:** Check api_server.py documentation
- **Extension Issues:** Review this guide and examples


