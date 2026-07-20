from collections import defaultdict
from decimal import Decimal, ROUND_HALF_UP
from typing import Any

from cortexweave_core.rag.pipelines.portfolio.planner import PortfolioQueryPlan


def execute_portfolio_plan(plan: PortfolioQueryPlan, context: dict[str, Any]) -> dict[str, Any]:
    dataset_results = {}
    warnings = []

    if plan.requires_external_metadata and plan.external_metadata_reason:
        warnings.append(plan.external_metadata_reason)

    for dataset_name in plan.datasets:
        dataset = context.get(dataset_name)
        if not isinstance(dataset, dict):
            continue

        rows = _dataset_rows(dataset, plan)
        rows = _filter_rows(rows, plan.filters)
        rows = _filter_numeric_rows(rows, plan.numeric_filters)
        if plan.group_by:
            rows = [
                row
                for row in rows
                if all(row.get(field) is not None for field in plan.group_by)
            ]
            if plan.sort:
                rows = _rank_rows_by_group(rows, plan.group_by, plan.sort)
            else:
                rows = _group_rows(rows, plan.group_by)
        rows = _sort_rows(rows, plan.sort)
        if plan.limit is not None:
            rows = rows[: plan.limit]

        dataset_results[dataset_name] = {
            "rows": rows,
            "totals": _totals(rows),
            "matched_rows": len(rows),
        }

    if not any(result["rows"] for result in dataset_results.values()):
        warnings.append("No structured portfolio rows matched the planned filters.")

    return {
        "plan": plan.to_dict(),
        "datasets": dataset_results,
        "warnings": warnings,
    }


def _dataset_rows(dataset: dict[str, Any], plan: PortfolioQueryPlan) -> list[dict[str, Any]]:
    if "holder_name" in plan.filters or "holder_name" in plan.group_by:
        holder_rows = dataset.get("holder_rows")
        if holder_rows:
            return list(holder_rows)
    rows = list(dataset.get("rows", []))
    return rows


def _filter_rows(
    rows: list[dict[str, Any]],
    filters: dict[str, str | list[str]],
) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if all(_matches_filter(row, key, value) for key, value in filters.items())
    ]


def _matches_filter(row: dict[str, Any], key: str, value: str | list[str]) -> bool:
    row_value = row.get(key)
    if row_value is None:
        return False
    values = value if isinstance(value, list) else [value]
    return any(
        str(row_value).strip().lower() == str(item).strip().lower()
        for item in values
    )


def _filter_numeric_rows(
    rows: list[dict[str, Any]],
    numeric_filters: dict[str, dict[str, float]],
) -> list[dict[str, Any]]:
    if not numeric_filters:
        return rows
    return [
        row
        for row in rows
        if all(_matches_numeric_filter(row, field, bounds) for field, bounds in numeric_filters.items())
    ]


def _matches_numeric_filter(row: dict[str, Any], field: str, bounds: dict[str, float]) -> bool:
    value = _decimal_or_none(row.get(field))
    if value is None:
        return False
    for operator, threshold in bounds.items():
        threshold_value = Decimal(str(threshold))
        if operator == "gt" and not value > threshold_value:
            return False
        if operator == "gte" and not value >= threshold_value:
            return False
        if operator == "lt" and not value < threshold_value:
            return False
        if operator == "lte" and not value <= threshold_value:
            return False
    return True


def _group_rows(rows: list[dict[str, Any]], group_by: list[str]) -> list[dict[str, Any]]:
    grouped = defaultdict(lambda: defaultdict(Decimal))
    metadata = {}

    for row in rows:
        key = tuple(row.get(field) for field in group_by)
        for field in group_by:
            metadata.setdefault(key, {})[field] = row.get(field)
        for field, value in row.items():
            if field in group_by:
                continue
            numeric = _decimal_or_none(value)
            if numeric is not None:
                grouped[key][field] += numeric

    output = []
    for key, amounts in grouped.items():
        row = dict(metadata.get(key, {}))
        for field, value in amounts.items():
            row[field] = _num(value)
        output.append(row)
    _recompute_allocation(output)
    return output


def _rank_rows_by_group(
    rows: list[dict[str, Any]],
    group_by: list[str],
    sort: dict[str, str],
) -> list[dict[str, Any]]:
    field = sort.get("field")
    if not field:
        return rows

    reverse = sort.get("direction") == "desc"
    ranked: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        value = _decimal_or_none(row.get(field))
        if value is None:
            continue
        key = tuple(row.get(group) for group in group_by)
        current = ranked.get(key)
        if current is None:
            ranked[key] = row
            continue
        current_value = _decimal_or_none(current.get(field))
        if current_value is None or (value > current_value if reverse else value < current_value):
            ranked[key] = row
    return list(ranked.values())


def _sort_rows(
    rows: list[dict[str, Any]],
    sort: dict[str, str] | None,
) -> list[dict[str, Any]]:
    if not sort:
        return rows
    field = sort.get("field")
    reverse = sort.get("direction") == "desc"
    populated = []
    missing = []
    for row in rows:
        value = _decimal_or_none(row.get(field))
        if value is None:
            missing.append(row)
        else:
            populated.append((value, row))
    sorted_rows = [
        row for _, row in sorted(populated, key=lambda item: item[0], reverse=reverse)
    ]
    return [*sorted_rows, *missing]


def _totals(rows: list[dict[str, Any]]) -> dict[str, float | None]:
    total_fields = (
        "current_value",
        "invested_value",
        "investment_amount",
        "cost",
        "market_value",
        "total_gain",
        "total_returns",
        "unrealized_gain_loss",
        "realized_gain_loss",
        "dividend_interest_paid",
        "interest_income",
    )
    totals = {}
    for field in total_fields:
        values = [_decimal_or_none(row.get(field)) for row in rows]
        values = [value for value in values if value is not None]
        if values:
            totals[field] = _num(sum(values, Decimal("0")))
    return totals


def _recompute_allocation(rows: list[dict[str, Any]]) -> None:
    total_current_value = sum(
        (_decimal_or_none(row.get("current_value")) or Decimal("0") for row in rows),
        Decimal("0"),
    )
    if not total_current_value:
        return
    for row in rows:
        current_value = _decimal_or_none(row.get("current_value")) or Decimal("0")
        row["current_allocation_percent"] = _num(
            (current_value / total_current_value * Decimal("100")).quantize(
                Decimal("0.01"),
                rounding=ROUND_HALF_UP,
            )
        )


def _decimal_or_none(value: Any) -> Decimal | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float, Decimal)):
        return Decimal(str(value))
    return None


def _num(value: Decimal | None) -> float | None:
    if value is None:
        return None
    return float(value)
