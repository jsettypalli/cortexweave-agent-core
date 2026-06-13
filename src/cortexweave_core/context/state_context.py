import json
import os


def _get_int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        parsed = int(raw)
        return parsed if parsed > 0 else default
    except ValueError:
        return default


def _get_set_env(name: str, default: set[str] | None = None) -> set[str]:
    raw = os.getenv(name)
    if raw is None:
        return set(default or set())
    return {value.strip() for value in raw.split(",") if value.strip()}


def _drop_empty_values(value):
    if value is None:
        return None
    if isinstance(value, str):
        return value if value.strip() else None
    if isinstance(value, dict):
        cleaned = {}
        for key, item in value.items():
            cleaned_item = _drop_empty_values(item)
            if cleaned_item is not None:
                cleaned[key] = cleaned_item
        return cleaned if cleaned else None
    if isinstance(value, list):
        cleaned = []
        for item in value:
            cleaned_item = _drop_empty_values(item)
            if cleaned_item is not None:
                cleaned.append(cleaned_item)
        return cleaned if cleaned else None
    return value


def _compact_value(value, max_str_len: int, max_items: int, depth: int = 0, max_depth: int = 6):
    if depth >= max_depth:
        return {"_truncated": True, "reason": "max_depth"}

    if isinstance(value, str):
        if len(value) <= max_str_len:
            return value
        return value[: max_str_len - 3] + "..."

    if isinstance(value, dict):
        items = list(value.items())
        trimmed_items = items[:max_items]
        compacted = {
            key: _compact_value(item, max_str_len, max_items, depth + 1, max_depth)
            for key, item in trimmed_items
        }
        if len(items) > max_items:
            compacted["_truncated"] = True
            compacted["_original_key_count"] = len(items)
        return compacted

    if isinstance(value, list):
        trimmed = value[:max_items]
        compacted_list = [
            _compact_value(item, max_str_len, max_items, depth + 1, max_depth)
            for item in trimmed
        ]
        if len(value) > max_items:
            compacted_list.append(
                {
                    "_truncated": True,
                    "_original_item_count": len(value),
                }
            )
        return compacted_list

    return value


def _is_blocked_key(key: str) -> bool:
    blocked_keys = _get_set_env("CW_CONTEXT_BLOCK_KEYS")
    blocked_prefixes = _get_set_env("CW_CONTEXT_BLOCK_PREFIXES", {"_cw_"})
    if key in blocked_keys:
        return True
    return any(key.startswith(prefix) for prefix in blocked_prefixes)


def _apply_filter(session_data: dict) -> dict:
    return {
        key: value
        for key, value in session_data.items()
        if key != "step_details" and not _is_blocked_key(key)
    }


def _fit_char_budget(context: dict, max_chars: int) -> dict:
    payload = json.dumps(context, separators=(",", ":"), default=str)
    if len(payload) <= max_chars:
        return context

    working = dict(context)
    removed_keys = []
    sorted_keys = sorted(
        working.keys(),
        key=lambda key: len(json.dumps(working[key], default=str)),
        reverse=True,
    )
    for key in sorted_keys:
        if len(json.dumps(working, separators=(",", ":"), default=str)) <= max_chars:
            break
        removed_keys.append(key)
        working.pop(key, None)

    if removed_keys:
        working["_context_truncated"] = {
            "removed_key_count": len(removed_keys),
            "removed_key_samples": removed_keys[:5],
        }
    return working


def build_customer_context_for_prompt(session_data: dict, step_details: dict | None) -> dict:
    """
    Build compact workflow context for LLM prompts without additional LLM calls.
    """
    if not isinstance(session_data, dict):
        return {}

    filtered = _apply_filter(session_data)
    max_str_len = _get_int_env("CW_CONTEXT_MAX_STR_LEN", 240)
    max_items = _get_int_env("CW_CONTEXT_MAX_ITEMS", 20)

    compacted = _compact_value(filtered, max_str_len=max_str_len, max_items=max_items)
    cleaned = _drop_empty_values(compacted) or {}
    max_chars = _get_int_env("CW_CONTEXT_MAX_JSON_CHARS", 4000)
    return _fit_char_budget(cleaned, max_chars)


def serialize_context_json(payload: dict) -> str:
    if not isinstance(payload, dict):
        return "{}"
    return json.dumps(payload, separators=(",", ":"), default=str)
