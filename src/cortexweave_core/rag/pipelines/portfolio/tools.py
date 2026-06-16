import asyncio
from collections.abc import Callable

from cortexweave_core.rag.db import get_session
from cortexweave_core.rag.pipelines.documents.dao import KnowledgeBaseDAO
from cortexweave_core.rag.pipelines.portfolio import analytics
from cortexweave_core.rag.pipelines.portfolio.dao import PortfolioReportDAO
from cortexweave_core.rag.pipelines.portfolio.models import PortfolioReport
from cortexweave_core.utils.config_loader import config


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
        filtered_rows.sort(key=lambda row: _numeric_sort_value(row, "total_gain"), reverse=True)
    elif normalized_sort in {"gain loss percent", "gain percent", "return percent", "percentage gain"}:
        filtered_rows.sort(key=lambda row: _numeric_sort_value(row, "gain_loss_percent"), reverse=True)
    elif normalized_sort in {"holding period months", "holding period", "held longest", "longest held", "longest", "held"}:
        filtered_rows.sort(key=lambda row: _numeric_sort_value(row, "holding_period_months"), reverse=True)
    elif normalized_sort in {"held shortest", "shortest held", "shortest"}:
        filtered_rows.sort(key=lambda row: _numeric_sort_value(row, "holding_period_months"))

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
