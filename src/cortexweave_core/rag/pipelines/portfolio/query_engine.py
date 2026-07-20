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
    answer = (
        _asset_class_segregation_answer(question, context)
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
    }


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
