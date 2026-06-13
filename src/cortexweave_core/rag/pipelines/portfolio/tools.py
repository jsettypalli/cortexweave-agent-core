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


async def get_mutual_fund_holdings() -> dict:
    """
    Returns individual mutual fund scheme holdings across all holders in the
    configured portfolio family's latest reports, including units, cost, market
    value, XIRR, gain/loss % and holding period.

    Returns:
        dict with "rows" (one per scheme per holder) and "totals".
    """
    return await asyncio.to_thread(_run, analytics.mutual_fund_holdings)


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
