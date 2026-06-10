import inspect
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from google.adk.tools.base_tool import BaseTool
from google.adk.tools.mcp_tool import MCPToolset, StreamableHTTPConnectionParams

try:
    from google.adk.tools.mcp_tool import SseConnectionParams
except Exception:
    SseConnectionParams = None

from .external_tool_request_runtime import decrypt_secret_record, is_explicit_env_reference, is_secret_ref, resolve_env_value


logger = logging.getLogger(__name__)


MCP_TOOLS_FILE_NAME = "mcp_tools.json"
MCP_TOOL_SECRETS_FILE_NAME = "mcp_tool_secrets.json"


class SanitizedMCPTool(BaseTool):
    def __init__(self, tool: BaseTool):
        super().__init__(
            name=tool.name,
            description=tool.description,
            is_long_running=getattr(tool, "is_long_running", False),
        )
        self._tool = tool

    def __getattr__(self, name: str) -> Any:
        return getattr(self._tool, name)

    def _get_declaration(self):
        return self._tool._get_declaration()

    async def run_async(self, *, args: dict[str, Any], tool_context) -> Any:
        result = await self._tool.run_async(args=args, tool_context=tool_context)
        return make_json_safe(result)


class SanitizingMCPToolset(MCPToolset):
    async def get_tools(self, readonly_context=None) -> List[BaseTool]:
        tools = await super().get_tools(readonly_context=readonly_context)
        return [SanitizedMCPTool(tool) for tool in tools]


def make_json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): make_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [make_json_safe(item) for item in value]

    value_type = type(value)
    if value_type.__module__.startswith("pydantic") and (value_type.__name__.lower().endswith("url") or hasattr(value, "_url")):
        return str(value)

    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            return make_json_safe(model_dump(mode="python"))
        except TypeError:
            pass
        except Exception:
            logger.debug("Failed to model_dump MCP value %r", value, exc_info=True)
        try:
            return make_json_safe(model_dump())
        except Exception:
            logger.debug("Failed to model_dump MCP value %r without mode", value, exc_info=True)

    dict_method = getattr(value, "dict", None)
    if callable(dict_method):
        try:
            return make_json_safe(dict_method())
        except Exception:
            logger.debug("Failed to dict-convert MCP value %r", value, exc_info=True)

    try:
        return make_json_safe(vars(value))
    except TypeError:
        return str(value)
    except Exception:
        logger.debug("Failed to inspect MCP value %r", value, exc_info=True)
        return str(value)


def load_mcp_secret_store(agent_dir: str | Path) -> Dict[str, Any]:
    target = Path(agent_dir) / "tools" / MCP_TOOL_SECRETS_FILE_NAME
    if not target.exists():
        return {"records": {}}
    try:
        raw = json.loads(target.read_text())
    except Exception:
        return {"records": {}}
    records = raw.get("records")
    if not isinstance(records, Mapping):
        records = {}
    return {"records": records}


def resolve_secret_value(raw: Any, secret_store: Optional[Mapping[str, Any]] = None) -> Optional[str]:
    if raw is None:
        return None
    if is_secret_ref(raw):
        secret_id = str(raw.get("secretId") or "").strip()
        records = (secret_store or {}).get("records") if isinstance(secret_store, Mapping) else None
        record = records.get(secret_id) if isinstance(records, Mapping) else None
        if not isinstance(record, Mapping):
            raise ValueError(f"Missing encrypted secret record for '{secret_id}'.")
        return decrypt_secret_record(record)
    if is_explicit_env_reference(raw):
        return resolve_env_value(raw)
    if isinstance(raw, str):
        return raw
    return str(raw)


def kv_list_to_dict(items: Any, *, secret_store: Optional[Mapping[str, Any]] = None) -> Dict[str, str]:
    output: Dict[str, str] = {}
    for item in items or []:
        if not isinstance(item, Mapping):
            continue
        key = str(item.get("key") or "").strip()
        if not key or item.get("enabled", True) is False:
            continue
        value = resolve_secret_value(item.get("value"), secret_store)
        output[key] = "" if value is None else str(value)
    return output


def _extract_servers(raw: Any) -> List[Dict[str, Any]]:
    if isinstance(raw, dict):
        servers = raw.get("servers")
    else:
        servers = raw
    if not isinstance(servers, list):
        return []
    return [item for item in servers if isinstance(item, dict)]


def _should_use_sse(server_def: Mapping[str, Any]) -> bool:
    transport = str(server_def.get("transport") or "streamable_http").strip().lower()
    server_url = str(server_def.get("serverUrl") or "").strip().lower()
    return transport == "sse" or server_url.endswith("/sse")


def _build_connection_params(server_def: Mapping[str, Any], secret_store: Mapping[str, Any]):
    headers = kv_list_to_dict(server_def.get("headers") or [], secret_store=secret_store)
    auth_type = str(server_def.get("authType") or "none").strip()
    auth_data = server_def.get("auth") or {}

    if auth_type == "bearer":
        token = resolve_secret_value(auth_data.get("bearerToken"), secret_store)
        if token:
            headers["Authorization"] = f"Bearer {token}"
    elif auth_type == "apiKey":
        api_key_name = str(auth_data.get("apiKeyName") or "").strip()
        api_key_value = resolve_secret_value(auth_data.get("apiKeyValue"), secret_store)
        if api_key_name and api_key_value:
            headers[api_key_name] = api_key_value

    timeout_ms = server_def.get("timeoutMs") or 15000
    timeout_seconds = float(timeout_ms) / 1000.0
    if _should_use_sse(server_def):
        if SseConnectionParams is None:
            raise RuntimeError("This ADK version does not expose SseConnectionParams.")
        kwargs = {
            "url": str(server_def.get("serverUrl") or ""),
            "headers": headers,
        }
        signature = inspect.signature(SseConnectionParams)
        if "timeout" in signature.parameters:
            kwargs["timeout"] = timeout_seconds
        if "sse_read_timeout" in signature.parameters:
            kwargs["sse_read_timeout"] = timeout_seconds
        elif "read_timeout_seconds" in signature.parameters:
            kwargs["read_timeout_seconds"] = timeout_seconds
        filtered_kwargs = {key: value for key, value in kwargs.items() if key in signature.parameters}
        return SseConnectionParams(**filtered_kwargs)

    kwargs = {
        "url": str(server_def.get("serverUrl") or ""),
        "headers": headers,
    }
    signature = inspect.signature(StreamableHTTPConnectionParams)
    if "timeout" in signature.parameters:
        kwargs["timeout"] = timeout_seconds
    elif "read_timeout_seconds" in signature.parameters:
        kwargs["read_timeout_seconds"] = timeout_seconds
    filtered_kwargs = {key: value for key, value in kwargs.items() if key in signature.parameters}
    return StreamableHTTPConnectionParams(**filtered_kwargs)


def _build_toolset(server_def: Mapping[str, Any], secret_store: Mapping[str, Any]):
    connection_params = _build_connection_params(server_def, secret_store)
    tool_filter = [str(item).strip() for item in (server_def.get("toolFilter") or []) if str(item).strip()]
    kwargs = {"connection_params": connection_params}
    signature = inspect.signature(SanitizingMCPToolset)
    if tool_filter and "tool_filter" in signature.parameters:
        kwargs["tool_filter"] = tool_filter
    if "errlog" in signature.parameters:
        kwargs["errlog"] = None
    filtered_kwargs = {key: value for key, value in kwargs.items() if key in signature.parameters}
    return SanitizingMCPToolset(**filtered_kwargs)


def load_mcp_tools(agent_dir: str) -> List[Any]:
    path = Path(agent_dir) / "tools" / MCP_TOOLS_FILE_NAME
    if not path.exists():
        return []

    try:
        raw = json.loads(path.read_text())
    except Exception:
        logger.warning("Failed to parse MCP tools config at %s", path)
        return []

    secret_store = load_mcp_secret_store(agent_dir)
    output: List[Any] = []
    for server_def in _extract_servers(raw):
        if server_def.get("enabled", True) is False:
            continue
        try:
            output.append(_build_toolset(server_def, secret_store))
        except Exception as exc:
            logger.warning("Failed to load MCP server %s: %s", server_def.get("name") or server_def.get("id") or "unknown", exc)
    return output
