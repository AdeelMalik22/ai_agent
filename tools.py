import json
from datetime import datetime, timezone

import requests

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



def get_current_time() -> dict:
    now = datetime.now(timezone.utc)
    return {
        "iso_utc": now.isoformat(),
        "epoch": int(now.timestamp()),
    }


def run_tool(tool_name: str, raw_arguments: str) -> str:
    try:
        args = json.loads(raw_arguments) if raw_arguments else {}
    except json.JSONDecodeError as exc:
        return json.dumps({"error": f"Invalid tool arguments JSON: {exc}"})

    try:
        print(tool_name)
        if tool_name == "get_current_time":
            return json.dumps(get_current_time())
        if tool_name == "get_weather":
            return json.dumps(get_weather(**args))
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
