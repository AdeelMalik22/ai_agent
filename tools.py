import json
import os
from datetime import datetime, timezone
from pathlib import Path

import requests
from ddgs import DDGS

from system_prompt import AGENT_PROMPTS


def get_coordinates(city: str):
    """Return latitude and longitude for a city using Nominatim (OpenStreetMap)."""
    url = "https://nominatim.openstreetmap.org/search"
    params = {"q": city, "format": "json", "limit": 1}
    headers = {"User-Agent": "BedrockAgent/1.0 (+https://yourdomain.com)"}
    try:
        response = requests.get(url, params=params, headers=headers, timeout=5)
        response.raise_for_status()  # Raise HTTPError if bad status
        data = response.json()
        if not data:
            return None, None
        lat = float(data[0]["lat"])
        lon = float(data[0]["lon"])
        return lat, lon
    except Exception as e:
        print("Error fetching coordinates:", e)
        return None, None


def get_weather(city: str, unit: str = "celsius") -> dict:
    """Fetch current weather from Open-Meteo API."""

    # Map city names to latitude/longitude
    lat, lon = get_coordinates(city)
    if not lat or not lon:
        return {"error": f"Could not find coordinates for city: {city}"}

    # Call Open-Meteo API
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
    try:
        response = requests.get(url, timeout=5)
        data = response.json()
        temp_c = data["current_weather"]["temperature"]
    except Exception as e:
        return {"error": f"Failed to fetch weather: {e}"}

    if unit.lower() == "fahrenheit":
        temp = round((temp_c * 9 / 5) + 32, 1)
        return {"city": city, "unit": "fahrenheit", "temperature": temp, "source": "Open-Meteo"}

    return {"city": city, "unit": "celsius", "temperature": temp_c, "source": "Open-Meteo"}


def web_search(query: str, max_results: int = 5) -> dict:
    """
    Search the web using DuckDuckGo (free, no API key needed).

    Args:
        query: Search query string
        max_results: Maximum number of results to return (default: 5, max: 10)

    Returns:
        dict with search results including title, body, and URL
    """
    try:
        ddgs = DDGS()
        results = ddgs.text(query, max_results=min(max_results, 10))

        if not results:
            return {
                "status": "success",
                "query": query,
                "results": [],
                "message": "No results found",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        formatted_results = []
        for result in results:
            formatted_results.append({
                "title": result.get("title", ""),
                "body": result.get("body", ""),
                "url": result.get("href", "")
            })

        return {
            "status": "success",
            "query": query,
            "results_count": len(formatted_results),
            "results": formatted_results,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source": "DuckDuckGo"
        }
    except Exception as e:
        return {
            "status": "error",
            "query": query,
            "error": str(e),
            "message": f"Failed to search web: {str(e)}",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }


def get_current_time() -> dict:
    now = datetime.now(timezone.utc)
    return {
        "iso_utc": now.isoformat(),
        "epoch": int(now.timestamp()),
    }


# File I/O Tool Implementations

def _get_workspace_root() -> Path:
    """Get the workspace root from environment or current directory."""
    workspace_root = os.getenv("WORKSPACE_ROOT", os.getcwd())
    return Path(workspace_root).resolve()


def _validate_path(file_path: str, allow_system_access: bool = False) -> tuple[bool, str, Path]:
    """
    Validate and normalize file path.

    Args:
        file_path: Path to validate
        allow_system_access: If True, allow access to any file. If False, restrict to workspace root.

    Returns: (is_valid, error_message, normalized_path)
    """
    try:
        request_path = Path(file_path).resolve()

        # If system access is not allowed, restrict to workspace
        if not allow_system_access:
            workspace_root = _get_workspace_root()
            # Check if path is within workspace
            if not str(request_path).startswith(str(workspace_root)):
                return False, f"Access denied: Path outside workspace root ({workspace_root})", request_path

        return True, "", request_path
    except Exception as e:
        return False, f"Invalid path: {str(e)}", Path(file_path)


def _get_max_file_size() -> int:
    """Get max file size in bytes (from KB env var)."""
    return int(os.getenv("MAX_FILE_SIZE", "100")) * 1024  # default 100KB


def _is_readable_extension(file_path: str) -> bool:
    """Check if file extension is allowed for reading."""
    allowed = {".py", ".ts", ".js", ".json", ".md", ".txt", ".yaml", ".yml", ".toml", ".env", ".sh", ".css", ".html", ".xml", ".sql", ".r", ".rb", ".go", ".java", ".cpp", ".c", ".h", ".java", ".cs"}
    ext = Path(file_path).suffix.lower()
    return ext in allowed or ext == ""  # allow no extension


def _is_writable_extension(file_path: str) -> bool:
    """Check if file extension is allowed for writing."""
    allowed = {".py", ".ts", ".js", ".json", ".md", ".txt", ".yaml", ".yml", ".toml", ".env", ".sh", ".css", ".html", ".xml", ".sql", ".r", ".rb", ".go", ".java", ".cpp", ".c", ".h", ".cs"}
    ext = Path(file_path).suffix.lower()
    return ext in allowed or ext == ""  # allow no extension


def read_file(file_path: str, allow_system_access: bool = False) -> dict:
    """
    Read and return file contents.

    Args:
        file_path: Path to file (relative to workspace root or absolute if allow_system_access=True)
        allow_system_access: If True, allow reading from any file on system

    Returns:
        dict with "content" key, or "error" key on failure
    """
    is_valid, error_msg, normalized_path = _validate_path(file_path, allow_system_access)
    if not is_valid:
        return {"error": error_msg}

    # Check extension
    if not _is_readable_extension(file_path):
        return {"error": f"File type not allowed for reading: {Path(file_path).suffix}"}

    # Check if file exists
    if not normalized_path.exists():
        return {"error": f"File not found: {file_path}"}

    # Check if it's a file (not directory)
    if not normalized_path.is_file():
        return {"error": f"Path is not a file: {file_path}"}

    # Check file size
    file_size = normalized_path.stat().st_size
    max_size = _get_max_file_size()
    if file_size > max_size:
        return {"error": f"File too large: {file_size} bytes (max: {max_size} bytes)"}

    try:
        content = normalized_path.read_text(encoding="utf-8")
        return {
            "success": True,
            "file_path": str(file_path),
            "size_bytes": file_size,
            "content": content
        }
    except UnicodeDecodeError:
        return {"error": f"File is not UTF-8 text: {file_path}"}
    except PermissionError:
        return {"error": f"Permission denied reading file: {file_path}"}
    except Exception as e:
        return {"error": f"Failed to read file: {str(e)}"}


def write_file(file_path: str, content: str) -> dict:
    """
    Write content to file. Creates file if it doesn't exist.
    Creates parent directories if needed.

    Args:
        file_path: Path to file (relative to workspace root)
        content: Content to write

    Returns:
        dict with success status
    """
    is_valid, error_msg, normalized_path = _validate_path(file_path)
    if not is_valid:
        return {"error": error_msg}

    # Check extension
    if not _is_writable_extension(file_path):
        return {"error": f"File type not allowed for writing: {Path(file_path).suffix}"}

    try:
        # Create parent directories
        normalized_path.parent.mkdir(parents=True, exist_ok=True)

        # Write file
        normalized_path.write_text(content, encoding="utf-8")

        return {
            "success": True,
            "file_path": str(file_path),
            "size_bytes": len(content.encode("utf-8")),
            "message": f"File written successfully: {file_path}"
        }
    except PermissionError:
        return {"error": f"Permission denied writing file: {file_path}"}
    except Exception as e:
        return {"error": f"Failed to write file: {str(e)}"}


def list_files(directory_path: str = ".") -> dict:
    """
    List contents of directory.

    Args:
        directory_path: Path to directory (relative to workspace root)

    Returns:
        dict with list of files and directories
    """
    is_valid, error_msg, normalized_path = _validate_path(directory_path)
    if not is_valid:
        return {"error": error_msg}

    # Check if path exists
    if not normalized_path.exists():
        return {"error": f"Directory not found: {directory_path}"}

    # Check if it's a directory
    if not normalized_path.is_dir():
        return {"error": f"Path is not a directory: {directory_path}"}

    try:
        entries = []
        for item in sorted(normalized_path.iterdir()):
            if item.is_file():
                size = item.stat().st_size
                entries.append({
                    "name": item.name,
                    "type": "file",
                    "size_bytes": size,
                    "path": str(item.relative_to(_get_workspace_root()))
                })
            elif item.is_dir():
                entries.append({
                    "name": item.name,
                    "type": "directory",
                    "path": str(item.relative_to(_get_workspace_root()))
                })

        return {
            "success": True,
            "directory": str(directory_path),
            "count": len(entries),
            "entries": entries
        }
    except PermissionError:
        return {"error": f"Permission denied reading directory: {directory_path}"}
    except Exception as e:
        return {"error": f"Failed to list directory: {str(e)}"}


def run_tool(tool_name: str, raw_arguments: str, allow_system_read: bool = False) -> str:
    try:
        args = json.loads(raw_arguments) if raw_arguments else {}
    except json.JSONDecodeError as exc:
        return json.dumps({"error": f"Invalid tool arguments JSON: {exc}"})

    try:
        print(f"[tool] {tool_name}")
        if tool_name == "get_current_time":
            return json.dumps(get_current_time())
        if tool_name == "get_weather":
            return json.dumps(get_weather(**args))
        if tool_name == "web_search":
            return json.dumps(web_search(**args))
        if tool_name == "read_file":
            return json.dumps(read_file(args.get("file_path", ""), allow_system_access=allow_system_read))
        if tool_name == "write_file":
            return json.dumps(write_file(**args))
        if tool_name == "list_files":
            return json.dumps(list_files(**args))
        return json.dumps({"error": f"Unknown tool: {tool_name}"})
    except Exception as exc:  # noqa: BLE001
        return json.dumps({"error": f"Tool execution failed: {exc}"})



TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Get current UTC time",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "City name",
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature unit",
                    },
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for current information, latest news, recent events, product releases, documentation, and real-time data. Use this for any factual question that requires up-to-date information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query - what you want to find information about",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results to return (default: 5, max: 10)",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file from the workspace. Supports: .py, .ts, .js, .json, .md, .txt, .yaml, .yml, .toml, .env, .sh, and common code files.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to file relative to workspace root (e.g., 'package.json', 'src/main.py', 'config/settings.yaml'). Use forward slashes.",
                    },
                },
                "required": ["file_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write or modify a file in the workspace. Creates parent directories automatically. Supports: .py, .ts, .js, .json, .md, .txt, .yaml, .yml, .toml, .env, .sh, and common code files.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to file relative to workspace root (e.g., 'package.json', 'src/main.py'). Use forward slashes.",
                    },
                    "content": {
                        "type": "string",
                        "description": "File content to write",
                    },
                },
                "required": ["file_path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List files and directories in a workspace directory",
            "parameters": {
                "type": "object",
                "properties": {
                    "directory_path": {
                        "type": "string",
                        "description": "Path to directory relative to workspace root (default: '.' for workspace root). Use forward slashes.",
                    },
                },
                "required": [],
            },
        },
    },
]


def execute_handoff(raw_arguments: str, active_agent: str, handoffs_this_turn: int, max_handoffs: int) -> tuple[str, str, int]:
    try:
        args = json.loads(raw_arguments) if raw_arguments else {}
    except json.JSONDecodeError as exc:
        return json.dumps({"error": f"Invalid handoff arguments JSON: {exc}"}), active_agent, handoffs_this_turn

    target_agent = args.get("target_agent")
    reason = args.get("reason", "")

    if not target_agent:
        return json.dumps({"error": "Missing required argument: target_agent"}), active_agent, handoffs_this_turn

    if target_agent not in AGENT_PROMPTS:
        return json.dumps({"error": f"Unknown target agent: {target_agent}"}), active_agent, handoffs_this_turn

    if handoffs_this_turn >= max_handoffs:
        return (
            json.dumps({"error": f"Handoff limit reached ({max_handoffs}) for this turn", "active_agent": active_agent}),
            active_agent,
            handoffs_this_turn,
        )

    if target_agent == active_agent:
        return json.dumps({"ok": True, "active_agent": active_agent, "message": "Already on requested agent"}), active_agent, handoffs_this_turn

    return (
        json.dumps({"ok": True, "active_agent": target_agent, "reason": reason}),
        target_agent,
        handoffs_this_turn + 1,
    )




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
                    "enum": list(AGENT_PROMPTS.keys()),
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
