import asyncio
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


def build_portfolio_context(reports: list[PortfolioReport]) -> dict[str, Any]:
    return {
        "asset_allocations": analytics.family_asset_allocation(reports),
        "holder_asset_allocations": analytics.holder_asset_allocation(reports),
        "sub_asset_allocations": analytics.family_sub_asset_allocation(reports),
        "mutual_fund_holdings": analytics.mutual_fund_holdings(reports),
        "holder_returns": analytics.holder_returns(reports),
        "pms_holdings": analytics.pms_analysis(reports),
        "bond_holdings": analytics.bond_analysis(reports),
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
    schema = portfolio_schema_from_context(context)
    plan = build_portfolio_query_plan(question, schema)
    data = execute_portfolio_plan(plan, context)
    warnings = data.pop("warnings", [])
    answer = _structured_fallback_answer(question, data)

    return {
        "answer": answer,
        "intent": plan.intent,
        "data": data,
        "sources": analytics.report_sources(reports),
        "warnings": warnings,
    }


def _structured_fallback_answer(question: str, data: dict) -> str:
    plan = data.get("plan", {})
    sort = plan.get("sort") or {}
    group_by = plan.get("group_by") or []
    metrics = plan.get("metrics") or []
    datasets = data.get("datasets", {})
    for dataset_name, dataset in datasets.items():
        rows = dataset.get("rows") or []
        if not rows:
            continue
        if group_by:
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
        metric_answer = _metric_lookup_answer(rows[0], metrics, dataset_name)
        if metric_answer:
            return metric_answer
        return f"Found {len(rows)} matching portfolio row(s) for: {question}"
    return "I don't have enough structured portfolio information to answer the question."


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
