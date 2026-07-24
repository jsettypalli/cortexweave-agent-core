import asyncio
from collections.abc import Callable

from cortexweave_core.rag.db import get_session
from cortexweave_core.rag.pipelines.documents.dao import KnowledgeBaseDAO
from cortexweave_core.rag.pipelines.portfolio import analytics
from cortexweave_core.rag.pipelines.portfolio.dao import PortfolioReportDAO
from cortexweave_core.rag.pipelines.portfolio.models import PortfolioReport
from cortexweave_core.rag.pipelines.portfolio.query_engine import (
    answer_portfolio_query as _answer_portfolio_query,
)
from cortexweave_core.utils.config_loader import config

DEFAULT_DIRECT_TOOL_ROW_LIMIT = 10
MAX_DIRECT_TOOL_ROW_LIMIT = 100
DEFAULT_DIRECT_TOOL_NESTED_LIMIT = 10
MAX_DIRECT_TOOL_NESTED_LIMIT = 25


def _config_or_raise(key: str) -> str:
    value = config.get(key)
    if not value:
        raise RuntimeError(f"{key} is not configured for this sub-agent")
    return value


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


def _not_found(knowledge_base_name: str, family_id: str) -> dict:
    return {
        "error": (
            f"No active portfolio reports found for family '{family_id}' "
            f"in knowledge base '{knowledge_base_name}'."
        )
    }


def _run(analytics_fn: Callable[[list[PortfolioReport]], dict]) -> dict:
    reports, knowledge_base_name, family_id = _load_reports()
    if not reports:
        return _not_found(knowledge_base_name, family_id)
    return analytics_fn(reports)


def _bounded_limit(value: int | None, default: int, maximum: int) -> int:
    if value is None or value <= 0:
        return default
    return min(value, maximum)


def _compact_sequence(values: object, limit: int) -> list:
    if not isinstance(values, list):
        return []
    return values[:limit]


def _compact_fund_overlap_result(
    result: dict,
    *,
    row_limit: int,
    stock_limit: int,
) -> dict:
    rows = list(result.get("rows") or [])
    compact_rows = []
    for row in rows[:row_limit]:
        compact = dict(row)
        compact["stocks"] = _compact_sequence(compact.get("stocks"), stock_limit)
        compact_rows.append(compact)

    totals = dict(result.get("totals") or {})
    totals["total_rows"] = len(rows)
    totals["returned_rows"] = len(compact_rows)
    totals["stocks_returned_per_pair"] = stock_limit

    warnings = list(result.get("warnings") or [])
    if len(rows) > len(compact_rows):
        warnings.append(
            f"Direct tool output is capped to {len(compact_rows)} rows out of {len(rows)} total matches."
        )
    if any(isinstance(row.get("stocks"), list) and len(row["stocks"]) > stock_limit for row in rows[:row_limit]):
        warnings.append(
            f"Shared stock lists are capped to top {stock_limit} by portfolio exposure per fund pair."
        )

    return {"rows": compact_rows, "totals": totals, "warnings": warnings}


def _compact_stock_overlap_result(
    result: dict,
    *,
    row_limit: int,
    fund_limit: int,
) -> dict:
    rows = list(result.get("rows") or [])
    compact_rows = []
    for row in rows[:row_limit]:
        compact = dict(row)
        compact["funds"] = _compact_sequence(compact.get("funds"), fund_limit)
        compact_rows.append(compact)

    totals = dict(result.get("totals") or {})
    totals["total_rows"] = len(rows)
    totals["returned_rows"] = len(compact_rows)
    totals["funds_returned_per_stock"] = fund_limit

    warnings = list(result.get("warnings") or [])
    if len(rows) > len(compact_rows):
        warnings.append(
            f"Direct tool output is capped to {len(compact_rows)} rows out of {len(rows)} total matches."
        )
    if any(isinstance(row.get("funds"), list) and len(row["funds"]) > fund_limit for row in rows[:row_limit]):
        warnings.append(
            f"Fund lists are capped to top {fund_limit} entries per stock in direct tool output."
        )

    return {"rows": compact_rows, "totals": totals, "warnings": warnings}


async def answer_portfolio_query(question: str) -> dict:
    """
    Answers portfolio questions using the deterministic portfolio query engine.

    Use this as the primary portfolio tool. Pass the latest user portfolio
    question exactly as written. Do not rewrite it or add holder/family qualifiers
    from chat history; the query engine handles planning, filtering, sorting,
    grouping, and ranking.

    If the returned dict contains `tables`, return the complete dict to the user
    as raw JSON only. Do not convert portfolio tables to Markdown; the UI renders
    structured tables in a modal.
    """
    return await _answer_portfolio_query(question)


def _normalize_text(value: str | None) -> str:
    if not value:
        return ""
    return " ".join(
        token
        for token in "".join(
            char.lower() if char.isalnum() else " "
            for char in value
        ).split()
        if token not in {"mr", "mrs", "ms", "shri", "smt"}
    )


def _matches_holder(row_holder_name: str | None, requested_holder_name: str | None) -> bool:
    requested = _normalize_text(requested_holder_name)
    if not requested:
        return True
    holder = _normalize_text(row_holder_name)
    requested_tokens = requested.split()
    return all(token in holder.split() for token in requested_tokens)


def _matches_text(row_value: str | None, requested_value: str | None) -> bool:
    requested = _normalize_text(requested_value)
    if not requested:
        return True
    return _normalize_text(row_value) == requested


def _numeric_sort_value(row: dict, key: str) -> float:
    value = row.get(key)
    if value is None:
        return float("-inf")
    return float(value)


def _sort_rows(rows: list[dict], key: str, *, reverse: bool = True) -> None:
    populated = [row for row in rows if row.get(key) is not None]
    missing = [row for row in rows if row.get(key) is None]
    populated.sort(key=lambda row: float(row.get(key)), reverse=reverse)
    rows[:] = [*populated, *missing]


def _top_total_gain_by_holder_asset_class(rows: list[dict]) -> list[dict]:
    top_rows: dict[tuple[str, str], dict] = {}
    for row in rows:
        holder_name = row.get("holder_name") or ""
        asset_class = row.get("asset_class") or ""
        key = (holder_name, asset_class)
        current = top_rows.get(key)
        if current is None or _numeric_sort_value(row, "total_gain") > _numeric_sort_value(current, "total_gain"):
            top_rows[key] = row

    return [
        {
            "holder_name": row.get("holder_name"),
            "asset_class": row.get("asset_class"),
            "scheme_name": row.get("scheme_name"),
            "sub_asset_class": row.get("sub_asset_class"),
            "total_gain": row.get("total_gain"),
            "gain_loss_percent": row.get("gain_loss_percent"),
            "market_value": row.get("market_value"),
        }
        for row in sorted(
            top_rows.values(),
            key=lambda item: ((item.get("holder_name") or ""), (item.get("asset_class") or "")),
        )
    ]


def _top_by_holder(rows: list[dict], field: str, output_field: str) -> list[dict]:
    top_rows: dict[str, dict] = {}
    for row in rows:
        holder_name = row.get("holder_name") or ""
        current = top_rows.get(holder_name)
        if current is None or _numeric_sort_value(row, field) > _numeric_sort_value(current, field):
            top_rows[holder_name] = row

    return [
        {
            "holder_name": row.get("holder_name"),
            "asset_class": row.get("asset_class"),
            "scheme_name": row.get("scheme_name"),
            "sub_asset_class": row.get("sub_asset_class"),
            output_field: row.get(field),
            "total_gain": row.get("total_gain"),
            "gain_loss_percent": row.get("gain_loss_percent"),
            "market_value": row.get("market_value"),
        }
        for row in sorted(top_rows.values(), key=lambda item: item.get("holder_name") or "")
    ]


async def get_asset_allocation() -> dict:
    """
    Returns the family-level asset allocation (e.g. Equity, Debt, Cash) across all
    holders in the configured portfolio family's latest reports, including
    per-asset-class current value, invested value, gains, XIRR, and allocation %.

    Returns:
        dict with "rows" (per asset class), "holder_rows" (per holder per asset
        class), and "totals" (aggregated current/invested value and gain).
    """
    return await asyncio.to_thread(_run, analytics.family_asset_allocation)


async def get_holder_asset_allocation() -> dict:
    """
    Returns each holder's individual asset allocation breakdown (Equity, Debt, etc.)
    from the configured portfolio family's latest reports.

    Returns:
        dict with "rows" (one per holder per asset class) and "totals".
    """
    return await asyncio.to_thread(_run, analytics.holder_asset_allocation)


async def get_sub_asset_allocation() -> dict:
    """
    Returns the family-level sub-asset-class allocation (e.g. Large Cap, Liquid
    Funds) across all holders in the configured portfolio family's latest reports.

    Returns:
        dict with "rows" (per asset class / sub-asset class), "holder_rows"
        (per holder), and "totals".
    """
    return await asyncio.to_thread(_run, analytics.family_sub_asset_allocation)


async def get_mutual_fund_holdings(
    holder_name: str | None = None,
    asset_class: str | None = None,
    sub_asset_class: str | None = None,
    sort_by: str | None = None,
    limit: int | None = None,
) -> dict:
    """
    Returns individual mutual fund scheme holdings across all holders in the
    configured portfolio family's latest reports, including units, cost, market
    value, XIRR, gain/loss % and holding period.

    Args:
        holder_name: Optional holder filter. For example, "Sulemanji Roowala"
            matches report holder "SULEMANJI M ROOWALA".
        asset_class: Optional exact broad asset class filter, such as "Equity",
            "Debt/Fixed Income", "Hybrid", or "Cash/Cash Equivalents".
        sub_asset_class: Optional exact sub-asset-class filter, such as
            "Large Cap", "Flexi Cap", or "Mid Cap".
        sort_by: Optional ranking field. Use "total_gain", "total_returns", or
            "highest_gain" for highest absolute gain. Use "holding_period_months",
            "held_longest", or "longest_held" for longest-held funds. Use
            "held_shortest" or "shortest_held" for shortest-held funds. Use
            "xirr_percent" or "highest_xirr" for highest XIRR ranking, and
            "lowest_xirr" or "worst_xirr" for lowest XIRR ranking. Use
            "gain_loss_percent" only for percentage-return ranking.
        limit: Optional maximum number of rows to return after filtering/sorting.

    Returns:
        dict with "rows" (one per scheme per holder), "totals", applied
        "filters", "top_total_gain_by_holder_asset_class", and
        "top_holding_period_by_holder" to support holder-specific ranking
        questions.
    """
    result = await asyncio.to_thread(_run, analytics.mutual_fund_holdings)
    rows = list(result.get("rows") or [])

    filtered_rows = [
        row for row in rows
        if _matches_holder(row.get("holder_name"), holder_name)
        and _matches_text(row.get("asset_class"), asset_class)
        and _matches_text(row.get("sub_asset_class"), sub_asset_class)
    ]

    normalized_sort = _normalize_text(sort_by)
    if normalized_sort in {"highest gain", "total gain", "total returns", "gain"}:
        _sort_rows(filtered_rows, "total_gain", reverse=True)
    elif normalized_sort in {"xirr percent", "xirr", "highest xirr", "best xirr", "highest return", "best return"}:
        _sort_rows(filtered_rows, "xirr_percent", reverse=True)
    elif normalized_sort in {"lowest xirr", "worst xirr", "lowest return", "worst return"}:
        _sort_rows(filtered_rows, "xirr_percent", reverse=False)
    elif normalized_sort in {"gain loss percent", "gain percent", "return percent", "percentage gain"}:
        _sort_rows(filtered_rows, "gain_loss_percent", reverse=True)
    elif normalized_sort in {"holding period months", "holding period", "held longest", "longest held", "longest", "held"}:
        _sort_rows(filtered_rows, "holding_period_months", reverse=True)
    elif normalized_sort in {"held shortest", "shortest held", "shortest", "held least", "least held"}:
        _sort_rows(filtered_rows, "holding_period_months", reverse=False)

    if limit is not None and limit > 0:
        filtered_rows = filtered_rows[:limit]

    return {
        "filters": {
            "holder_name": holder_name,
            "asset_class": asset_class,
            "sub_asset_class": sub_asset_class,
            "sort_by": sort_by,
            "limit": limit,
        },
        "top_total_gain_by_holder_asset_class": _top_total_gain_by_holder_asset_class(rows),
        "top_holding_period_by_holder": _top_by_holder(rows, "holding_period_months", "holding_period_months"),
        "rows": filtered_rows,
        "totals": result.get("totals"),
    }


async def get_holder_returns() -> dict:
    """
    Returns each holder's overall portfolio return (the "Grand Total" row from
    their latest report), including current value, invested value, gain/loss and XIRR.

    Returns:
        dict with "rows" (one per holder) and "totals".
    """
    return await asyncio.to_thread(_run, analytics.holder_returns)


async def get_pms_holdings() -> dict:
    """
    Returns Portfolio Management Service (PMS) holdings across all holders in the
    configured portfolio family's latest reports.

    Returns:
        dict with "rows" (one per PMS scheme per holder) and "totals".
    """
    return await asyncio.to_thread(_run, analytics.pms_analysis)


async def get_bond_holdings() -> dict:
    """
    Returns bond holdings across all holders in the configured portfolio family's
    latest reports, including rating, maturity date, market value and accrued
    interest income.

    Returns:
        dict with "rows" (one per bond per holder) and "totals".
    """
    return await asyncio.to_thread(_run, analytics.bond_analysis)


async def get_report_sources() -> dict:
    """
    Returns the source portfolio report(s) (file name, holder, report date) that
    back the data returned by the other portfolio tools.

    Returns:
        dict with "sources": list of {"source_file_name", "holder_name", "report_date"}.
    """
    reports, knowledge_base_name, family_id = await asyncio.to_thread(_load_reports)
    if not reports:
        return {"sources": [], **_not_found(knowledge_base_name, family_id)}
    return {"sources": analytics.report_sources(reports)}


async def get_fund_resolution_status() -> dict:
    """
    Returns mutual-fund enrichment resolution status for the configured
    portfolio family, including unresolved/ambiguous funds and candidate matches.
    """
    return await asyncio.to_thread(_run, analytics.fund_resolution_status)


async def get_fund_stock_holdings(
    holder_name: str | None = None,
    scheme_name: str | None = None,
    stock_name: str | None = None,
    sector: str | None = None,
    nature: str | None = None,
) -> dict:
    """
    Returns underlying stock/security holdings fetched through fund enrichment.

    Use this when the user asks what stocks are inside mutual funds, what a
    specific fund holds, or what fund-level exposure exists to a stock/sector.
    """
    result = await asyncio.to_thread(_run, analytics.fund_stock_holdings)
    rows = [
        row for row in result.get("rows") or []
        if _matches_holder(row.get("holder_name"), holder_name)
        and _matches_contains(row.get("scheme_name"), scheme_name, row.get("matched_name"))
        and _matches_contains(row.get("stock_name"), stock_name)
        and _matches_contains(row.get("sector"), sector)
        and _matches_text(row.get("nature"), nature)
    ]
    return {"filters": {
        "holder_name": holder_name,
        "scheme_name": scheme_name,
        "stock_name": stock_name,
        "sector": sector,
        "nature": nature,
    }, "rows": rows, "totals": result.get("totals")}


async def get_stock_overlap(
    min_fund_count: int = 2,
    limit: int | None = None,
    fund_limit: int | None = None,
) -> dict:
    """
    Returns stocks duplicated across enriched mutual funds, with aggregate
    portfolio-weighted exposure and contributing funds.

    Prefer `answer_portfolio_query` for user-facing answers. This direct tool is
    compacted by default to avoid sending large overlap tables to the LLM.
    """
    reports, knowledge_base_name, family_id = await asyncio.to_thread(_load_reports)
    if not reports:
        return _not_found(knowledge_base_name, family_id)
    result = analytics.stock_overlap(reports, min_fund_count=max(1, min_fund_count))
    return _compact_stock_overlap_result(
        result,
        row_limit=_bounded_limit(limit, DEFAULT_DIRECT_TOOL_ROW_LIMIT, MAX_DIRECT_TOOL_ROW_LIMIT),
        fund_limit=_bounded_limit(fund_limit, DEFAULT_DIRECT_TOOL_NESTED_LIMIT, MAX_DIRECT_TOOL_NESTED_LIMIT),
    )


async def get_sector_overlap() -> dict:
    """
    Returns sector exposure aggregated from enriched underlying fund holdings.
    """
    return await asyncio.to_thread(_run, analytics.sector_overlap)


async def get_fund_overlap_matrix(limit: int | None = None, stock_limit: int | None = None) -> dict:
    """
    Returns pairwise mutual-fund overlap based on shared underlying stocks.

    Prefer `answer_portfolio_query` for user-facing answers. This direct tool
    returns only a compact preview by default; totals indicate how many matches
    exist in the full matrix.
    """
    result = await asyncio.to_thread(_run, analytics.fund_overlap_matrix)
    return _compact_fund_overlap_result(
        result,
        row_limit=_bounded_limit(limit, DEFAULT_DIRECT_TOOL_ROW_LIMIT, MAX_DIRECT_TOOL_ROW_LIMIT),
        stock_limit=_bounded_limit(stock_limit, DEFAULT_DIRECT_TOOL_NESTED_LIMIT, MAX_DIRECT_TOOL_NESTED_LIMIT),
    )


def _matches_contains(row_value: str | None, requested_value: str | None, alternate_value: str | None = None) -> bool:
    requested = _normalize_text(requested_value)
    if not requested:
        return True
    values = [_normalize_text(row_value), _normalize_text(alternate_value)]
    return any(requested in value or value in requested for value in values if value)
