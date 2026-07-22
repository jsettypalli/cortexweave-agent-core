import asyncio
import re
from typing import Any

from cortexweave_core.rag.db import get_session
from cortexweave_core.rag.pipelines.documents.dao import KnowledgeBaseDAO
from cortexweave_core.rag.pipelines.portfolio import analytics
from cortexweave_core.rag.pipelines.portfolio.dao import PortfolioReportDAO
from cortexweave_core.rag.pipelines.portfolio.executor import execute_portfolio_plan
from cortexweave_core.rag.pipelines.portfolio.models import PortfolioReport
from cortexweave_core.rag.pipelines.portfolio.planner import (
    build_portfolio_query_plan,
    portfolio_schema_from_context,
)
from cortexweave_core.utils.config_loader import config

MAX_OVERLAP_ROWS = 50


def build_portfolio_context(reports: list[PortfolioReport]) -> dict[str, Any]:
    return {
        "asset_allocations": analytics.family_asset_allocation(reports),
        "holder_asset_allocations": analytics.holder_asset_allocation(reports),
        "sub_asset_allocations": analytics.family_sub_asset_allocation(reports),
        "mutual_fund_holdings": analytics.mutual_fund_holdings(reports),
        "holder_returns": analytics.holder_returns(reports),
        "pms_holdings": analytics.pms_analysis(reports),
        "bond_holdings": analytics.bond_analysis(reports),
        "fund_resolution_status": analytics.fund_resolution_status(reports),
        "fund_stock_holdings": analytics.fund_stock_holdings(reports),
        "stock_overlap": analytics.stock_overlap(reports),
        "sector_overlap": analytics.sector_overlap(reports),
        "fund_overlap_matrix": analytics.fund_overlap_matrix(reports),
    }


def _config_or_raise(key: str) -> str:
    value = config.get(key)
    if not value:
        raise RuntimeError(f"{key} is not configured for this sub-agent")
    return str(value)


def _load_reports() -> tuple[list[PortfolioReport], str, str]:
    knowledge_base_name = _config_or_raise("RAG_KNOWLEDGE_BASE_NAME")
    family_id = _config_or_raise("RAG_FAMILY_ID")
    session = get_session()
    try:
        kb = KnowledgeBaseDAO.find_by_name(session, knowledge_base_name)
        if not kb:
            return [], knowledge_base_name, family_id
        reports = PortfolioReportDAO.latest_reports_for_family(session, kb.id, family_id)
        return reports, knowledge_base_name, family_id
    finally:
        session.close()


async def answer_portfolio_query(question: str) -> dict:
    """
    Answer a portfolio question through the deterministic portfolio query engine.

    This is the primary portfolio tool for generated agents. It uses the configured
    RAG_KNOWLEDGE_BASE_NAME and RAG_FAMILY_ID, builds a structured plan, and
    executes filtering, sorting, grouping, and ranking deterministically.
    """
    return await asyncio.to_thread(_answer_portfolio_query_sync, question)


def _answer_portfolio_query_sync(question: str) -> dict:
    reports, knowledge_base_name, family_id = _load_reports()
    if not reports:
        return {
            "answer": (
                f"No active portfolio reports found for family '{family_id}' "
                f"in knowledge base '{knowledge_base_name}'."
            ),
            "intent": "not_found",
            "data": {"rows": []},
            "sources": [],
            "warnings": [],
        }

    context = build_portfolio_context(reports)
    enrichment_answer = _enrichment_answer(
        question,
        context,
        knowledge_base_name=knowledge_base_name,
        family_id=family_id,
    )
    if enrichment_answer:
        return {
            **enrichment_answer,
            "sources": analytics.report_sources(reports),
        }
    schema = portfolio_schema_from_context(context)
    plan = build_portfolio_query_plan(question, schema)
    data = execute_portfolio_plan(plan, context)
    warnings = data.pop("warnings", [])
    answer = (
        _asset_class_segregation_answer(question, context)
        or
        _sub_asset_parent_percent_answer(question, plan.to_dict(), context)
        or
        _sub_asset_parent_allocation_answer(question, plan.to_dict(), data, context)
        or _structured_fallback_answer(question, data)
    )

    return {
        "answer": answer,
        "intent": plan.intent,
        "data": data,
        "sources": analytics.report_sources(reports),
        "warnings": warnings,
        "tables": [],
    }


def _enrichment_answer(
    question: str,
    context: dict[str, Any],
    knowledge_base_name: str | None = None,
    family_id: str | None = None,
) -> dict[str, Any] | None:
    normalized = question.lower()
    if any(term in normalized for term in ("unresolved", "ambiguous", "could not resolve", "fund match")):
        dataset = context.get("fund_resolution_status") or {"rows": []}
        rows = dataset.get("rows") or []
        issues = [row for row in rows if row.get("status") in {"ambiguous", "not_found", "failed"}]
        return {
            "answer": f"Found {len(issues)} fund resolution issue(s).",
            "intent": "resolution_status",
            "data": {"datasets": {"fund_resolution_status": {"rows": issues, "matched_rows": len(issues)}}},
            "warnings": _enrichment_warnings(rows),
            "tables": [_table(
                "fund_resolution_status",
                "Fund Resolution Issues",
                "review",
                [
                    ("scheme_name", "Statement Fund", "text"),
                    ("statement_category", "Category", "text"),
                    ("status", "Status", "text"),
                    ("matched_name", "Best Match", "text"),
                    ("score", "Score", "number"),
                    ("candidates", "Candidates", "list"),
                ],
                issues,
                actions=["resolve_match"],
            )],
        }
    if any(term in normalized for term in ("sector", "industry", "concentration")) and any(term in normalized for term in ("overlap", "exposure", "concentration", "duplicated", "duplicate")):
        dataset = context.get("sector_overlap") or {"rows": []}
        rows = dataset.get("rows") or []
        return {
            "answer": f"Found sector exposure across {len(rows)} sector(s).",
            "intent": "sector_overlap",
            "data": {"datasets": {"sector_overlap": {"rows": rows, "matched_rows": len(rows)}}},
            "warnings": _enrichment_warnings((context.get("fund_resolution_status") or {}).get("rows") or []),
            "tables": [_table(
                "sector_overlap",
                "Sector Exposure",
                "modal",
                [
                    ("sector", "Sector", "text"),
                    ("fund_count", "Funds", "number"),
                    ("aggregate_portfolio_exposure_percent", "Portfolio Exposure", "percent"),
                    ("aggregate_weighted_market_value", "Weighted Value", "currency"),
                    ("top_contributing_funds", "Top Contributing Funds", "list"),
                ],
                rows,
                default_sort={"key": "aggregate_portfolio_exposure_percent", "direction": "desc"},
                actions=["csv_export"],
            )],
        }
    if any(term in normalized for term in ("matrix", "pairwise", "fund overlap")):
        dataset = context.get("fund_overlap_matrix") or {"rows": []}
        all_rows = dataset.get("rows") or []
        rows = all_rows[:MAX_OVERLAP_ROWS]
        return {
            "answer": _limited_overlap_answer(
                len(all_rows),
                "fund pair(s) with shared underlying stocks",
                rows,
            ),
            "intent": "fund_overlap_matrix",
            "data": {"datasets": {"fund_overlap_matrix": {"rows": rows, "matched_rows": len(rows), "total_rows": len(all_rows)}}},
            "warnings": _enrichment_warnings((context.get("fund_resolution_status") or {}).get("rows") or []),
            "tables": [_table(
                "fund_overlap_matrix",
                "Fund Overlap Matrix",
                "matrix",
                [
                    ("fund_a", "Fund A", "text"),
                    ("fund_b", "Fund B", "text"),
                    ("shared_stocks", "Shared Stocks", "number"),
                    ("stocks", "Stocks", "list"),
                ],
                rows,
                default_sort={"key": "shared_stocks", "direction": "desc"},
                actions=["csv_export"],
                warnings=_limit_warnings(len(all_rows), len(rows)),
            )],
        }
    if _asks_for_stock_overlap(normalized):
        requested_holders = _requested_holder_names(context, normalized)
        equity_only = _stock_overlap_equity_only(normalized)
        all_rows = _stock_overlap_rows(context, requested_holders, equity_only=equity_only)
        requested_limit = _requested_limit(normalized)
        row_limit = requested_limit or MAX_OVERLAP_ROWS
        rows = all_rows[:row_limit]
        can_load_full_table = requested_limit is None
        holder_label = f" for {', '.join(requested_holders)}" if requested_holders else ""
        return {
            "answer": _limited_overlap_answer(
                len(all_rows),
                f"duplicated stock(s) across enriched mutual funds{holder_label}",
                rows,
            ),
            "intent": "stock_overlap",
            "data": {
                "datasets": {
                    "stock_overlap": {
                        "rows": rows,
                        "matched_rows": len(rows),
                        "total_rows": len(all_rows),
                        "holder_names": requested_holders,
                        "nature": "EQUITY" if equity_only else None,
                    }
                }
            },
            "warnings": _enrichment_warnings((context.get("fund_resolution_status") or {}).get("rows") or []),
            "tables": [_table(
                "stock_overlap",
                "Duplicated Stock Exposure",
                "modal",
                [
                    ("stock_name", "Stock", "text"),
                    ("sector", "Sector", "text"),
                    ("nature", "Nature", "text"),
                    ("fund_count", "Funds", "number"),
                    ("aggregate_portfolio_exposure_percent", "Portfolio Exposure", "percent"),
                    ("aggregate_weighted_market_value", "Weighted Value", "currency"),
                    ("max_pct_in_single_fund", "Max In Single Fund", "percent"),
                    ("funds", "Funds Holding It", "list"),
                ],
                rows,
                default_sort={"key": "aggregate_portfolio_exposure_percent", "direction": "desc"},
                actions=["csv_export"],
                warnings=(
                    _table_scope_warnings(len(all_rows), len(rows))
                    if can_load_full_table
                    else _limit_warnings(len(all_rows), len(rows))
                ),
                total_rows=len(all_rows),
                query_ref=_query_ref(question, knowledge_base_name, family_id) if can_load_full_table else None,
            )],
        }
    return None


def _asks_for_stock_overlap(normalized: str) -> bool:
    stock_terms = ("stock", "stocks", "security", "securities", "constituents", "holdings")
    overlap_terms = (
        "overlap",
        "duplicate",
        "duplicated",
        "common",
        "same",
        "underlying",
        "more than one",
        "multiple funds",
        "across funds",
        "across my mutual funds",
    )
    return any(term in normalized for term in stock_terms) and any(term in normalized for term in overlap_terms)


def _stock_overlap_equity_only(normalized: str) -> bool:
    broad_terms = ("security", "securities", "instrument", "instruments", "all underlying", "all holdings")
    return not any(term in normalized for term in broad_terms)


def _requested_limit(normalized: str) -> int | None:
    match = re.search(r"\btop\s+(\d{1,3})\b", normalized)
    if not match:
        return None
    return max(1, min(int(match.group(1)), MAX_OVERLAP_ROWS))


def _stock_overlap_rows(
    context: dict[str, Any],
    holder_names: list[str],
    equity_only: bool = True,
) -> list[dict[str, Any]]:
    exposure_rows = (context.get("fund_stock_holdings") or {}).get("rows") or []
    if holder_names:
        exposure_rows = [row for row in exposure_rows if row.get("holder_name") in holder_names]
    if equity_only:
        exposure_rows = [row for row in exposure_rows if row.get("nature") == "EQUITY"]
    if not exposure_rows:
        return []

    grouped: dict[str, dict[str, Any]] = {}
    for exposure in exposure_rows:
        stock = exposure.get("stock_name")
        if not stock:
            continue
        target = grouped.setdefault(str(stock), {
            "stock_name": stock,
            "sector": exposure.get("sector"),
            "nature": exposure.get("nature"),
            "funds": [],
            "aggregate_portfolio_exposure_percent": 0.0,
            "aggregate_weighted_market_value": 0.0,
            "max_pct_in_single_fund": 0.0,
        })
        fund = exposure.get("matched_name") or exposure.get("scheme_name")
        if fund and fund not in target["funds"]:
            target["funds"].append(fund)
        target["aggregate_portfolio_exposure_percent"] += float(exposure.get("portfolio_weighted_pct") or 0)
        target["aggregate_weighted_market_value"] += float(exposure.get("weighted_market_value") or 0)
        target["max_pct_in_single_fund"] = max(
            target["max_pct_in_single_fund"],
            float(exposure.get("pct_of_fund_assets") or 0),
        )

    rows = []
    for row in grouped.values():
        row["fund_count"] = len(row["funds"])
        if row["fund_count"] >= 2:
            row["aggregate_portfolio_exposure_percent"] = round(row["aggregate_portfolio_exposure_percent"], 4)
            row["aggregate_weighted_market_value"] = round(row["aggregate_weighted_market_value"], 2)
            rows.append(row)
    rows.sort(key=lambda row: row["aggregate_portfolio_exposure_percent"], reverse=True)
    return rows


def _table(
    table_id: str,
    title: str,
    display: str,
    columns: list[tuple[str, str, str]],
    rows: list[dict[str, Any]],
    default_sort: dict[str, str] | None = None,
    actions: list[str] | None = None,
    warnings: list[str] | None = None,
    total_rows: int | None = None,
    query_ref: dict[str, str] | None = None,
) -> dict[str, Any]:
    return {
        "id": table_id,
        "title": title,
        "display": display,
        "columns": [{"key": key, "label": label, "type": type_} for key, label, type_ in columns],
        "rows": rows,
        "default_sort": default_sort,
        "actions": actions or [],
        "warnings": warnings or [],
        "total_rows": total_rows if total_rows is not None else len(rows),
        "query_ref": query_ref,
    }


def _query_ref(question: str, knowledge_base_name: str | None, family_id: str | None) -> dict[str, str] | None:
    if not knowledge_base_name or not family_id:
        return None
    return {"text": question, "knowledge_base_name": knowledge_base_name, "family_id": family_id}


def _limited_overlap_answer(total_rows: int, label: str, rows: list[dict[str, Any]]) -> str:
    if total_rows > len(rows):
        return f"Found {total_rows} {label}. Showing the top {len(rows)} by portfolio exposure."
    return f"Found {total_rows} {label}."


def _limit_warnings(total_rows: int, shown_rows: int = MAX_OVERLAP_ROWS) -> list[str]:
    if total_rows <= shown_rows:
        return []
    return [f"Showing top {shown_rows} rows by exposure out of {total_rows} total matches."]


def _table_scope_warnings(total_rows: int, highlighted_rows: int) -> list[str]:
    if total_rows <= highlighted_rows:
        return []
    return [f"Answer highlights top {highlighted_rows} rows by exposure; open table to load and search all {total_rows} matches."]


def _enrichment_warnings(rows: list[dict[str, Any]]) -> list[str]:
    issue_count = sum(1 for row in rows if row.get("status") in {"ambiguous", "not_found", "failed"})
    if not issue_count:
        return []
    return [f"{issue_count} fund(s) could not be fully enriched, so overlap results may be incomplete."]


def _asset_class_segregation_answer(question: str, context: dict[str, Any]) -> str | None:
    normalized = question.lower()
    if "asset class segregation" not in normalized and not ("segregation" in normalized and "asset class" in normalized):
        return None

    sub_asset_data = context.get("sub_asset_allocations") or {}
    family_rows = _segregation_rows(sub_asset_data.get("rows") or [])
    holder_rows = _segregation_rows(sub_asset_data.get("holder_rows") or [])
    if not family_rows and not holder_rows:
        return None

    sections = []
    if family_rows:
        sections.append("Family Asset Class Segregation: " + _segregation_rows_text(family_rows) + ".")
    if holder_rows:
        by_holder: dict[str, list[dict[str, Any]]] = {}
        for row in holder_rows:
            holder = row.get("holder_name")
            if holder:
                by_holder.setdefault(str(holder), []).append(row)
        if by_holder:
            holder_parts = [
                f"{holder}: {_segregation_rows_text(rows)}"
                for holder, rows in sorted(by_holder.items())
            ]
            sections.append("Individual Asset Class Segregation: " + "; ".join(holder_parts) + ".")
    return " ".join(sections) if sections else None


def _segregation_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if row.get("asset_bucket") in {"Debt", "Equity"}
        and isinstance(row.get("current_value"), (int, float))
        and row.get("current_value") != 0
    ]


def _segregation_rows_text(rows: list[dict[str, Any]]) -> str:
    labels = []
    for row in sorted(rows, key=lambda item: (str(item.get("asset_bucket")), str(item.get("asset_class")), str(item.get("sub_asset_class")))):
        asset_class = row.get("asset_class")
        sub_asset_class = row.get("sub_asset_class")
        current_value = row.get("current_value")
        allocation = row.get("current_allocation_percent")
        name = " / ".join(str(part) for part in (asset_class, sub_asset_class) if part)
        if not name:
            continue
        if allocation is not None:
            labels.append(f"{name}: current_value {current_value}, allocation {allocation}%")
        else:
            labels.append(f"{name}: current_value {current_value}")
    return "; ".join(labels)


def _sub_asset_parent_allocation_answer(
    question: str,
    plan: dict[str, Any],
    data: dict[str, Any],
    context: dict[str, Any],
) -> str | None:
    normalized = question.lower()
    if not all(term in normalized for term in ("allocation", "overall")):
        return None
    if not any(term in normalized for term in ("debt portion", "debt allocation", "debt part")):
        return None

    filters = plan.get("filters") or {}
    sub_asset_class = filters.get("sub_asset_class")
    if not isinstance(sub_asset_class, str):
        return None

    sub_rows = (data.get("datasets", {}).get("sub_asset_allocations") or {}).get("rows") or []
    if len(sub_rows) != 1:
        return None
    sub_row = sub_rows[0]
    sub_value = sub_row.get("current_value")
    parent_asset_class = sub_row.get("asset_class")
    if not isinstance(sub_value, (int, float)) or not parent_asset_class:
        return None

    holder_name = filters.get("holder_name")
    parent_row = _matching_parent_asset_row(context, str(parent_asset_class), holder_name)
    if not parent_row:
        return None
    parent_value = parent_row.get("current_value")
    if not isinstance(parent_value, (int, float)) or parent_value == 0:
        return None

    parent_percent = round(sub_value / parent_value * 100, 2)
    overall_percent = sub_row.get("current_allocation_percent")
    if not isinstance(overall_percent, (int, float)):
        overall_total = _overall_current_value(context, holder_name)
        if not overall_total:
            return None
        overall_percent = round(sub_value / overall_total * 100, 2)

    holder_prefix = f"For {holder_name}, " if isinstance(holder_name, str) else ""
    return (
        f"{holder_prefix}{sub_asset_class} is {parent_percent}% of {parent_asset_class} "
        f"and {overall_percent}% of the overall portfolio."
    )


def _sub_asset_parent_percent_answer(
    question: str,
    plan: dict[str, Any],
    context: dict[str, Any],
) -> str | None:
    normalized = question.lower()
    if "allocation" not in normalized:
        return None
    if not any(term in normalized for term in ("%", "percent", "percentage", "terms")):
        return None
    if not any(term in normalized for term in ("equity", "debt", "hybrid", "asset")):
        return None

    sub_asset_class = _requested_sub_asset_class(plan, context, normalized)
    parent_asset_class = _requested_parent_asset_class(plan, context, normalized, sub_asset_class)
    if not sub_asset_class or not parent_asset_class:
        return None

    holder_names = _requested_holder_names(context, normalized)
    if holder_names:
        rows = []
        for holder_name in holder_names:
            sub_row = _matching_sub_asset_row(context, sub_asset_class, parent_asset_class, holder_name)
            parent_row = _matching_parent_asset_row(context, parent_asset_class, holder_name)
            if sub_row and parent_row:
                rows.append(_sub_asset_percent_row(context, sub_row, parent_row, holder_name))
    else:
        sub_row = _matching_sub_asset_row(context, sub_asset_class, parent_asset_class, None)
        parent_row = _matching_parent_asset_row(context, parent_asset_class, None)
        rows = [_sub_asset_percent_row(context, sub_row, parent_row, None)] if sub_row and parent_row else []

    rows = [row for row in rows if row is not None]
    if not rows:
        return None

    show_overall = any(term in normalized for term in ("total asset", "total assets", "overall", "portfolio"))
    parts = []
    for row in rows:
        holder_prefix = f"{row['holder_name']}: " if row.get("holder_name") else ""
        text = (
            f"{holder_prefix}{sub_asset_class} is {row['parent_percent']}% "
            f"of {parent_asset_class}"
        )
        if show_overall:
            text += f" and {row['overall_percent']}% of overall assets"
        text += f" (current value {row['current_value']})."
        parts.append(text)
    return " ".join(parts)


def _matching_parent_asset_row(
    context: dict[str, Any],
    asset_class: str,
    holder_name: Any,
) -> dict[str, Any] | None:
    if isinstance(holder_name, str):
        rows = (context.get("asset_allocations") or {}).get("holder_rows") or []
        return next(
            (
                row
                for row in rows
                if row.get("holder_name") == holder_name and row.get("asset_class") == asset_class
            ),
            None,
        )

    rows = (context.get("asset_allocations") or {}).get("rows") or []
    return next((row for row in rows if row.get("asset_class") == asset_class), None)


def _matching_sub_asset_row(
    context: dict[str, Any],
    sub_asset_class: str,
    parent_asset_class: str,
    holder_name: Any,
) -> dict[str, Any] | None:
    dataset = context.get("sub_asset_allocations") or {}
    rows = dataset.get("holder_rows") if isinstance(holder_name, str) else dataset.get("rows")
    return next(
        (
            row
            for row in rows or []
            if row.get("sub_asset_class") == sub_asset_class
            and row.get("asset_class") == parent_asset_class
            and (not isinstance(holder_name, str) or row.get("holder_name") == holder_name)
        ),
        None,
    )


def _sub_asset_percent_row(
    context: dict[str, Any],
    sub_row: dict[str, Any] | None,
    parent_row: dict[str, Any] | None,
    holder_name: Any,
) -> dict[str, Any] | None:
    if not sub_row or not parent_row:
        return None
    sub_value = sub_row.get("current_value")
    parent_value = parent_row.get("current_value")
    if not isinstance(sub_value, (int, float)) or not isinstance(parent_value, (int, float)) or parent_value == 0:
        return None
    overall_total = _overall_current_value(context, holder_name)
    if not overall_total:
        return None
    return {
        "holder_name": holder_name if isinstance(holder_name, str) else None,
        "current_value": sub_value,
        "parent_percent": round(sub_value / parent_value * 100, 2),
        "overall_percent": round(sub_value / overall_total * 100, 2),
    }


def _requested_sub_asset_class(
    plan: dict[str, Any],
    context: dict[str, Any],
    normalized: str,
) -> str | None:
    filters = plan.get("filters") or {}
    planned = filters.get("sub_asset_class")
    if isinstance(planned, str):
        return planned
    rows = (context.get("sub_asset_allocations") or {}).get("rows") or []
    return _find_named_row_value(rows, "sub_asset_class", normalized)


def _requested_parent_asset_class(
    plan: dict[str, Any],
    context: dict[str, Any],
    normalized: str,
    sub_asset_class: str | None,
) -> str | None:
    filters = plan.get("filters") or {}
    planned = filters.get("asset_class")
    if isinstance(planned, str):
        return planned
    rows = (context.get("asset_allocations") or {}).get("rows") or []
    explicit = _find_named_row_value(rows, "asset_class", normalized)
    if explicit:
        return explicit
    if sub_asset_class:
        sub_rows = (context.get("sub_asset_allocations") or {}).get("rows") or []
        match = next((row for row in sub_rows if row.get("sub_asset_class") == sub_asset_class), None)
        if match and isinstance(match.get("asset_class"), str):
            return match["asset_class"]
    return None


def _requested_holder_names(context: dict[str, Any], normalized: str) -> list[str]:
    rows = (context.get("holder_returns") or {}).get("rows") or []
    holders = sorted({str(row.get("holder_name")) for row in rows if row.get("holder_name")})
    return [holder for holder in holders if _holder_name_requested(holder, normalized)]


def _holder_name_requested(holder_name: str, normalized: str) -> bool:
    tokens = _simple_tokens(holder_name)
    if not tokens:
        return False
    return any(token in _simple_tokens(normalized) for token in tokens if len(token) > 3)


def _find_named_row_value(
    rows: list[dict[str, Any]],
    field: str,
    normalized: str,
) -> str | None:
    normalized_tokens = set(_simple_tokens(normalized))
    best: tuple[int, str] | None = None
    for row in rows:
        value = row.get(field)
        if not isinstance(value, str):
            continue
        tokens = set(_simple_tokens(value))
        if not tokens or not tokens.issubset(normalized_tokens):
            continue
        score = len(tokens)
        if best is None or score > best[0]:
            best = (score, value)
    return best[1] if best else None


def _simple_tokens(value: str) -> list[str]:
    return [token for token in "".join(char.lower() if char.isalnum() else " " for char in value).split()]


def _overall_current_value(context: dict[str, Any], holder_name: Any) -> float | None:
    if isinstance(holder_name, str):
        rows = (context.get("asset_allocations") or {}).get("holder_rows") or []
        values = [
            row.get("current_value")
            for row in rows
            if row.get("holder_name") == holder_name and isinstance(row.get("current_value"), (int, float))
        ]
    else:
        values = [
            row.get("current_value")
            for row in (context.get("asset_allocations") or {}).get("rows") or []
            if isinstance(row.get("current_value"), (int, float))
        ]
    total = sum(values)
    return total or None


def _structured_fallback_answer(question: str, data: dict) -> str:
    plan = data.get("plan", {})
    sort = plan.get("sort") or {}
    group_by = plan.get("group_by") or []
    metrics = plan.get("metrics") or []
    datasets = data.get("datasets", {})
    allocation_answer = _asset_allocation_view_answer(question, datasets)
    if allocation_answer:
        return allocation_answer
    for dataset_name, dataset in datasets.items():
        rows = dataset.get("rows") or []
        if not rows:
            continue
        if group_by:
            grouped_ranking_answer = _grouped_ranking_answer(rows, group_by, sort, dataset_name)
            if grouped_ranking_answer:
                return grouped_ranking_answer
            ratio_answer = _debt_to_equity_ratio_answer(question, rows, group_by)
            if ratio_answer:
                return ratio_answer
            labels = []
            for row in rows:
                label = " / ".join(
                    str(row.get(field))
                    for field in group_by
                    if row.get(field) is not None
                )
                current_value = row.get("current_value")
                allocation = row.get("current_allocation_percent")
                if label and current_value is not None and allocation is not None:
                    labels.append(
                        f"{label}: current_value {current_value}, "
                        f"allocation {allocation}%"
                    )
                elif label and current_value is not None:
                    labels.append(f"{label}: current_value {current_value}")
            if labels:
                return "Distribution: " + "; ".join(labels) + "."
        if sort:
            field = sort.get("field")
            row = rows[0]
            if dataset_name == "holder_returns":
                name = row.get("holder_name")
            else:
                name = (
                    row.get("scheme_name")
                    or row.get("security_name")
                    or row.get("sub_asset_class")
                    or row.get("asset_class")
                    or row.get("holder_name")
                )
            value = row.get(field)
            direction = "lowest" if sort.get("direction") == "asc" else "highest"
            if name and field:
                return f"The {direction} matching result is {name} with {field} {value}."
        total_answer = _metric_total_answer(dataset, metrics)
        if total_answer:
            return total_answer
        average_answer = _average_metric_answer(question, rows, metrics)
        if average_answer:
            return average_answer
        metric_answer = _metric_lookup_answer(rows[0], metrics, dataset_name)
        if metric_answer:
            return metric_answer
        return f"Found {len(rows)} matching portfolio row(s) for: {question}"
    return "I don't have enough structured portfolio information to answer the question."


def _asset_allocation_view_answer(question: str, datasets: dict[str, Any]) -> str | None:
    normalized = question.lower()
    if "asset allocation" not in normalized and "allocation wise" not in normalized and "allocation-wise" not in normalized:
        return None

    family_rows = (datasets.get("asset_allocations") or {}).get("rows") or []
    holder_rows = (datasets.get("holder_asset_allocations") or {}).get("rows") or []
    if not family_rows and not holder_rows:
        return None

    sections = []
    if family_rows:
        sections.append("Family Allocation: " + _allocation_rows_text(family_rows) + ".")
    if holder_rows:
        by_holder: dict[str, list[dict[str, Any]]] = {}
        for row in holder_rows:
            holder = row.get("holder_name")
            if holder:
                by_holder.setdefault(str(holder), []).append(row)
        if by_holder:
            holder_parts = [
                f"{holder}: {_allocation_rows_text(rows)}"
                for holder, rows in sorted(by_holder.items())
            ]
            sections.append("Individual Allocation: " + "; ".join(holder_parts) + ".")
    return " ".join(sections) if sections else None


def _allocation_rows_text(rows: list[dict[str, Any]]) -> str:
    labels = []
    for row in rows:
        name = row.get("asset_class") or row.get("sub_asset_class") or row.get("asset_bucket")
        current_value = row.get("current_value")
        allocation = row.get("current_allocation_percent")
        if not name:
            continue
        if current_value is not None and allocation is not None:
            labels.append(f"{name}: current_value {current_value}, allocation {allocation}%")
        elif current_value is not None:
            labels.append(f"{name}: current_value {current_value}")
        elif allocation is not None:
            labels.append(f"{name}: allocation {allocation}%")
    return "; ".join(labels)


def _debt_to_equity_ratio_answer(
    question: str,
    rows: list[dict[str, Any]],
    group_by: list[str],
) -> str | None:
    normalized = question.lower()
    if "debt" not in normalized or "equity" not in normalized or "ratio" not in normalized:
        return None
    if "asset_bucket" not in group_by:
        return None

    by_bucket = {
        str(row.get("asset_bucket") or "").strip().lower(): row
        for row in rows
    }
    debt = by_bucket.get("debt")
    equity = by_bucket.get("equity")
    if not debt or not equity:
        return None

    debt_value = debt.get("current_value")
    equity_value = equity.get("current_value")
    if not isinstance(debt_value, (int, float)) or not isinstance(equity_value, (int, float)):
        return None
    if equity_value == 0:
        return None

    ratio_percent = round((debt_value / equity_value) * 100, 2)
    debt_allocation = debt.get("current_allocation_percent")
    equity_allocation = equity.get("current_allocation_percent")
    if isinstance(debt_allocation, (int, float)) and isinstance(equity_allocation, (int, float)):
        return (
            f"Debt is {debt_allocation}% and Equity is {equity_allocation}% of the portfolio. "
            f"The debt-to-equity ratio is {ratio_percent}%."
        )
    return f"The debt-to-equity ratio is {ratio_percent}%."


def _grouped_ranking_answer(
    rows: list[dict[str, Any]],
    group_by: list[str],
    sort: dict[str, str],
    dataset_name: str,
) -> str | None:
    field = sort.get("field")
    if not field:
        return None

    ranking_word = "highest" if sort.get("direction") == "desc" else "lowest"
    holding_word = "longest" if sort.get("direction") == "desc" else "shortest"
    labels = []
    for row in rows:
        group_label = " / ".join(
            str(row.get(group))
            for group in group_by
            if row.get(group) is not None
        )
        name = _row_display_name(row, dataset_name)
        value = row.get(field)
        if not group_label or not name or value is None:
            continue
        if field == "holding_period_months":
            labels.append(f"For {group_label}, {name} has been held {holding_word} at {value} months")
        elif field.endswith("_percent"):
            labels.append(f"For {group_label}, {name} has the {ranking_word} {field} at {value}%")
        else:
            labels.append(f"For {group_label}, {name} has the {ranking_word} {field} at {value}")
    if labels:
        return "; ".join(labels) + "."
    return None


def _metric_total_answer(dataset: dict[str, Any], metrics: list[str]) -> str | None:
    rows = dataset.get("rows") or []
    if len(rows) <= 1:
        return None
    totals = dataset.get("totals") or {}
    for metric in metrics:
        value = totals.get(metric)
        if value is not None:
            return f"total {metric} is {value}."
    return None


def _average_metric_answer(
    question: str,
    rows: list[dict[str, Any]],
    metrics: list[str],
) -> str | None:
    normalized = question.lower()
    if not any(term in normalized for term in ("average", "avg", "mean")):
        return None

    for metric in metrics:
        if metric == "xirr_percent":
            reported = _reported_xirr_answer(rows)
            if reported:
                return reported
        values = [
            row.get(metric)
            for row in rows
            if isinstance(row.get(metric), (int, float))
        ]
        if not values:
            continue
        average = round(sum(values) / len(values), 2)
        suffix = "%" if metric.endswith("_percent") else ""
        return f"Average {metric} across {len(values)} matching rows is {average}{suffix}."
    return None


def _reported_xirr_answer(rows: list[dict[str, Any]]) -> str | None:
    reported_rows = [
        row
        for row in rows
        if isinstance(row.get("xirr_percent"), (int, float))
        and row.get("asset_class")
        and not row.get("scheme_name")
        and not row.get("security_name")
    ]
    if len(reported_rows) != 1:
        return None

    row = reported_rows[0]
    parts = []
    holder = row.get("holder_name")
    asset_class = row.get("asset_class")
    if holder:
        parts.append(str(holder))
    if asset_class:
        parts.append(str(asset_class))
    label = " for " + " / ".join(parts) if parts else ""
    return f"PDF-reported XIRR{label} is {row['xirr_percent']}%."


def _metric_lookup_answer(
    row: dict[str, Any],
    metrics: list[str],
    dataset_name: str,
) -> str | None:
    for metric in metrics:
        if metric not in row:
            continue
        value = row.get(metric)
        if value is None:
            continue
        name = _row_display_name(row, dataset_name)
        if name:
            return f"{metric} for {name} is {value}."
        return f"{metric} is {value}."
    return None


def _row_display_name(row: dict[str, Any], dataset_name: str) -> str | None:
    if dataset_name == "holder_returns":
        return row.get("holder_name")
    return (
        row.get("scheme_name")
        or row.get("security_name")
        or row.get("sub_asset_class")
        or row.get("asset_class")
        or row.get("holder_name")
    )
