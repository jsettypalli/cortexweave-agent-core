import base64
import json
import os
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.keywrap import aes_key_unwrap
from google.cloud import kms


def _item_get(item: Any, key: str, default: Any = None) -> Any:
    if isinstance(item, Mapping):
        return item.get(key, default)
    return getattr(item, key, default)


def _b64decode(value: str) -> bytes:
    return base64.b64decode(value.encode("ascii"))


def is_explicit_env_reference(raw: Any) -> bool:
    if not isinstance(raw, str):
        return False
    trimmed = raw.strip()
    if not trimmed:
        return False
    if trimmed.startswith("env:"):
        return len(trimmed) > 4
    if trimmed.startswith("${") and trimmed.endswith("}"):
        return len(trimmed) > 3
    if trimmed.startswith("{{") and trimmed.endswith("}}"):
        return len(trimmed) > 4
    return False


def resolve_env_value(raw: Any) -> Optional[str]:
    if not isinstance(raw, str):
        return None
    trimmed = raw.strip()
    if not trimmed:
        return None
    if trimmed.startswith("env:"):
        key = trimmed[4:]
    elif trimmed.startswith("${") and trimmed.endswith("}"):
        key = trimmed[2:-1]
    elif trimmed.startswith("{{") and trimmed.endswith("}}"):
        key = trimmed[2:-2]
    else:
        key = trimmed
    return os.getenv(key)


def is_secret_ref(raw: Any) -> bool:
    return (
        isinstance(raw, Mapping)
        and str(raw.get("kind") or "").strip() == "secretRef"
        and bool(str(raw.get("secretId") or "").strip())
    )


def load_secret_store(agent_dir: str | Path) -> Dict[str, Any]:
    tools_path = Path(agent_dir) / "tools" / "external_tool_secrets.json"
    legacy_path = Path(agent_dir) / "external_tool_secrets.json"
    target = tools_path if tools_path.exists() else legacy_path
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


def _unwrap_dek(wrapped_dek: bytes, envelope: Mapping[str, Any]) -> bytes:
    if str(envelope.get("kekProvider") or "").strip() == "local-testing":
        if os.getenv("USE_LOCAL_ENVELOPE_ENCRYPTION", "").strip().lower() != "true":
            raise ValueError("Local envelope encryption is disabled for this runtime.")
        encoded_key = os.getenv("EXTERNAL_TOOLS_LOCAL_KEK", "").strip()
        if not encoded_key:
            raise ValueError("EXTERNAL_TOOLS_LOCAL_KEK is required when USE_LOCAL_ENVELOPE_ENCRYPTION=true.")
        wrapping_key = _b64decode(encoded_key)
        return aes_key_unwrap(wrapping_key, wrapped_dek)

    project_id = os.getenv("GCP_PROJECT_ID", "").strip()
    location = os.getenv("GCP_KMS_LOCATION", "").strip()
    key_ring = os.getenv("GCP_KMS_KEY_RING", "").strip()
    crypto_key = os.getenv("GCP_KMS_CRYPTO_KEY", "").strip()
    if not all([project_id, location, key_ring, crypto_key]):
        raise ValueError("GCP KMS is required but not fully configured for runtime decryption.")
    client = kms.KeyManagementServiceClient()
    key_name = client.crypto_key_path(project_id, location, key_ring, crypto_key)
    return client.decrypt(request={"name": key_name, "ciphertext": wrapped_dek}).plaintext


def decrypt_secret_record(record: Mapping[str, Any]) -> str:
    envelope = record.get("envelope") or {}
    wrapped_dek = _b64decode(str(envelope.get("wrappedDek") or ""))
    iv = _b64decode(str(envelope.get("iv") or ""))
    ciphertext = _b64decode(str(envelope.get("ciphertext") or ""))
    tag = _b64decode(str(envelope.get("tag") or ""))
    dek = _unwrap_dek(wrapped_dek, envelope)
    plaintext = AESGCM(dek).decrypt(iv, ciphertext + tag, None)
    return plaintext.decode("utf-8")


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
        key = str(_item_get(item, "key", "") or "").strip()
        if not key:
            continue
        if _item_get(item, "enabled", True) is False:
            continue
        value = resolve_secret_value(_item_get(item, "value"), secret_store)
        output[key] = "" if value is None else str(value)
    return output


def apply_path_params(url: str, items: Any, *, secret_store: Optional[Mapping[str, Any]] = None) -> str:
    updated = url
    for item in items or []:
        key = str(_item_get(item, "key", "") or "").strip()
        if not key:
            continue
        if _item_get(item, "enabled", True) is False:
            continue
        value = resolve_secret_value(_item_get(item, "value"), secret_store)
        updated = updated.replace(f"{{{key}}}", "" if value is None else str(value))
    return updated


def build_request_config(
    tool_def: Mapping[str, Any],
    *,
    path_params: Optional[dict] = None,
    params: Optional[dict] = None,
    headers: Optional[dict] = None,
    body: Optional[str] = None,
    agent_dir: Optional[str | Path] = None,
    secret_store: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    resolved_secret_store = secret_store if secret_store is not None else (
        load_secret_store(agent_dir) if agent_dir is not None else None
    )
    method = str(tool_def.get("method") or "GET").upper()
    merged_path_params = kv_list_to_dict(tool_def.get("pathParams") or [], secret_store=resolved_secret_store)
    merged_headers = kv_list_to_dict(tool_def.get("headers") or [], secret_store=resolved_secret_store)
    merged_params = kv_list_to_dict(tool_def.get("queryParams") or [], secret_store=resolved_secret_store)

    if path_params:
        merged_path_params.update({str(k): str(v) for k, v in path_params.items()})
    if headers:
        merged_headers.update({str(k): str(v) for k, v in headers.items()})
    if params:
        merged_params.update({str(k): str(v) for k, v in params.items()})

    url = apply_path_params(
        str(tool_def.get("url") or ""),
        [{"key": key, "value": value, "enabled": True} for key, value in merged_path_params.items()],
        secret_store=resolved_secret_store,
    )
    if "{" in url or "}" in url:
        return {"error": "Missing path params for URL template"}

    auth_type = str(tool_def.get("authType") or "none")
    auth_data = tool_def.get("auth") or {}
    auth = None
    if auth_type == "bearer":
        token = resolve_secret_value(auth_data.get("bearerToken"), resolved_secret_store)
        if not token:
            return {"error": "Missing bearer token"}
        merged_headers["Authorization"] = f"Bearer {token}"
    elif auth_type == "basic":
        username = resolve_secret_value(auth_data.get("username"), resolved_secret_store)
        password = resolve_secret_value(auth_data.get("password"), resolved_secret_store)
        if username is None or password is None:
            return {"error": "Missing basic auth credentials"}
        auth = (username, password)
    elif auth_type == "apiKey":
        api_key_name = str(auth_data.get("apiKeyName") or "").strip()
        api_key_value = resolve_secret_value(auth_data.get("apiKeyValue"), resolved_secret_store)
        api_key_location = str(auth_data.get("apiKeyLocation") or "header").strip()
        if not api_key_name or not api_key_value:
            return {"error": "Missing API key auth config"}
        if api_key_location == "query":
            merged_params[api_key_name] = api_key_value
        elif api_key_location == "cookie":
            cookie_value = merged_headers.get("Cookie")
            merged_headers["Cookie"] = f"{cookie_value}; {api_key_name}={api_key_value}" if cookie_value else f"{api_key_name}={api_key_value}"
        else:
            merged_headers[api_key_name] = api_key_value

    body_type = str(tool_def.get("bodyType") or "none")
    data = None
    json_body = None
    files = None

    if body is not None:
        if body_type == "json":
            try:
                json_body = json.loads(body)
            except Exception:
                return {"error": "Invalid JSON body supplied to tool"}
        elif body_type == "raw":
            data = body
        elif body_type in {"form", "form_urlencoded"}:
            try:
                parsed = json.loads(body)
            except Exception:
                return {"error": "Form body must be a JSON object string"}
            if not isinstance(parsed, dict):
                return {"error": "Form body must be a JSON object string"}
            data = parsed
        elif body_type == "multipart":
            try:
                parsed = json.loads(body)
            except Exception:
                return {"error": "Multipart body must be a JSON object string"}
            if not isinstance(parsed, dict):
                return {"error": "Multipart body must be a JSON object string"}
            files = [(str(key), (None, str(value))) for key, value in parsed.items()]
        else:
            data = body
    else:
        if body_type == "json":
            raw = str(tool_def.get("body") or "")
            if raw:
                try:
                    json_body = json.loads(raw)
                except Exception:
                    return {"error": "Invalid JSON body in tool definition"}
        elif body_type == "raw":
            data = str(tool_def.get("body") or "")
        elif body_type in {"form", "form_urlencoded"}:
            data = kv_list_to_dict(tool_def.get("bodyForm") or [], secret_store=resolved_secret_store)
        elif body_type == "multipart":
            form_fields = kv_list_to_dict(tool_def.get("bodyForm") or [], secret_store=resolved_secret_store)
            files = [(key, (None, value)) for key, value in form_fields.items()]

    return {
        "method": method,
        "url": url,
        "headers": merged_headers,
        "params": merged_params,
        "data": data,
        "json": json_body,
        "files": files,
        "auth": auth,
        "timeout": (tool_def.get("timeoutMs") or 15000) / 1000.0,
    }
