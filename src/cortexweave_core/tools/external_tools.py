import asyncio
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from .external_tool_request_runtime import apply_path_params, build_request_config, kv_list_to_dict, load_secret_store


def _normalize_tool_name(name: str) -> str:
    base = re.sub(r"[^a-zA-Z0-9]+", "_", (name or "").strip().lower())
    base = re.sub(r"_+", "_", base).strip("_")
    if not base:
        base = "external_tool"
    if not re.match(r"^[a-zA-Z_]", base):
        base = f"tool_{base}"
    return base


def _build_tool_doc(tool_def: Dict[str, Any], secret_store: Dict[str, Any]) -> str:
    method = (tool_def.get("method") or "GET").upper()
    url = apply_path_params(tool_def.get("url") or "", tool_def.get("pathParams") or [], secret_store=secret_store)
    description = tool_def.get("description") or ""
    headers = kv_list_to_dict(tool_def.get("headers") or [], secret_store=secret_store)
    path_params = kv_list_to_dict(tool_def.get("pathParams") or [], secret_store=secret_store)
    params = kv_list_to_dict(tool_def.get("queryParams") or [], secret_store=secret_store)
    body_type = tool_def.get("bodyType") or "none"
    return (
        f"{description}\n"
        f"External REST tool.\n"
        f"Method: {method}\n"
        f"URL: {url}\n"
        f"Path params: {path_params}\n"
        f"Default headers: {headers}\n"
        f"Default query params: {params}\n"
        f"Body type: {body_type}\n"
        "Parameters schema (optional):\n"
        "- path_params: dict[str, str] path params to override/extend\n"
        "- params: dict[str, str] query params to override/extend\n"
        "- headers: dict[str, str] headers to override/extend\n"
        "- body: str (JSON string for json body, raw text for raw body)\n"
        "Return schema:\n"
        "- status: int HTTP status code\n"
        "- headers: dict response headers\n"
        "- body: str response text\n"
        "- json: parsed JSON if available, else null\n"
    ).strip()


def _make_tool(tool_def: Dict[str, Any], tool_name: str, agent_dir: str):
    secret_store = load_secret_store(agent_dir)
    tool_doc = _build_tool_doc(tool_def, secret_store)

    async def _tool(
        tool_context,
        path_params: Optional[Dict[str, str]] = None,
        params: Optional[Dict[str, str]] = None,
        headers: Optional[Dict[str, str]] = None,
        body: Optional[str] = None,
    ):
        request_config = build_request_config(
            tool_def,
            path_params=path_params,
            params=params,
            headers=headers,
            body=body,
            agent_dir=agent_dir,
            secret_store=secret_store,
        )
        if "error" in request_config:
            return {"status": None, "error": request_config["error"]}

        def _call():
            response = requests.request(
                request_config["method"],
                request_config["url"],
                headers=request_config["headers"],
                params=request_config["params"],
                data=request_config["data"],
                json=request_config["json"],
                files=request_config["files"],
                auth=request_config["auth"],
                timeout=request_config["timeout"],
            )
            try:
                body_json = response.json()
            except Exception:
                body_json = None
            return {
                "status": response.status_code,
                "headers": dict(response.headers),
                "body": response.text,
                "json": body_json,
            }

        return await asyncio.to_thread(_call)

    _tool.__name__ = tool_name
    _tool.__qualname__ = tool_name
    _tool.__doc__ = tool_doc
    _tool.__annotations__ = {
        "path_params": Optional[Dict[str, str]],
        "params": Optional[Dict[str, str]],
        "headers": Optional[Dict[str, str]],
        "body": Optional[str],
        "return": dict,
    }
    return _tool


def load_external_tools(agent_dir: str) -> List[Any]:
    path = Path(agent_dir) / "tools" / "external_tools.json"
    legacy_path = Path(agent_dir) / "external_tools.json"
    if not path.exists() and legacy_path.exists():
        path = legacy_path
    if not path.exists():
        return []

    try:
        raw = json.loads(path.read_text())
    except Exception:
        return []

    if isinstance(raw, dict):
        tools = raw.get("tools")
    else:
        tools = raw
    if not isinstance(tools, list):
        return []

    seen = set()
    output = []
    for tool_def in tools:
        if not isinstance(tool_def, dict):
            continue
        name = _normalize_tool_name(tool_def.get("name") or tool_def.get("id") or "external_tool")
        base = name
        i = 1
        while name in seen:
            i += 1
            name = f"{base}_{i}"
        seen.add(name)
        output.append(_make_tool(tool_def, name, agent_dir))
    return output
