from collections import defaultdict
from decimal import Decimal, ROUND_HALF_UP
import re
from typing import Iterable

from cortexweave_core.rag.pipelines.portfolio.models import PortfolioReport
from cortexweave_core.rag.pipelines.portfolio.mutual_fund_value_extractor import extract_mutual_fund_values


def family_asset_allocation(reports: list[PortfolioReport]) -> dict:
    rows_by_asset = defaultdict(lambda: _empty_amount_row())
    holder_rows = []
    for report in reports:
        for row in report.asset_allocations:
            if row.asset_class == "Grand Total":
                continue
            target = rows_by_asset[row.asset_class]
            _add_amounts(target, row)
            holder_rows.append(_allocation_row_to_dict(row, report.holder_name))
    rows = _with_recomputed_allocations(rows_by_asset)
    return {
        "rows": rows,
        "holder_rows": holder_rows,
        "totals": _totals(rows),
    }


def holder_asset_allocation(reports: list[PortfolioReport]) -> dict:
    rows = []
    for report in reports:
        for row in report.asset_allocations:
            if row.asset_class != "Grand Total":
                rows.append(_allocation_row_to_dict(row, report.holder_name))
    return {"rows": rows, "totals": _totals(rows)}


def family_sub_asset_allocation(reports: list[PortfolioReport]) -> dict:
    rows_by_key = defaultdict(lambda: _empty_amount_row())
    holder_rows = []
    for report in reports:
        for row in report.sub_asset_allocations:
            key = (row.asset_class, row.sub_asset_class)
            target = rows_by_key[key]
            target["asset_class"] = row.asset_class
            target["sub_asset_class"] = row.sub_asset_class
            _add_amounts(target, row)
            holder_rows.append(_sub_allocation_row_to_dict(row, report.holder_name))
    rows = _with_recomputed_allocations(rows_by_key.values())
    return {"rows": rows, "holder_rows": holder_rows, "totals": _totals(rows)}


def holder_returns(reports: list[PortfolioReport]) -> dict:
    rows = []
    for report in reports:
        grand_total = next((row for row in report.asset_allocations if row.asset_class == "Grand Total"), None)
        if not grand_total:
            continue
        rows.append(_allocation_row_to_dict(grand_total, report.holder_name))
    return {"rows": rows, "totals": _totals(rows)}


def pms_analysis(reports: list[PortfolioReport]) -> dict:
    rows = []
    for report in reports:
        for holding in report.pms_holdings:
            rows.append({
                "holder_name": report.holder_name,
                "scheme_name": holding.scheme_name,
                "investment_date": holding.investment_date.isoformat() if holding.investment_date else None,
                "investment_amount": _num(holding.investment_amount),
                "current_value": _num(holding.current_value),
                "valuation_date": holding.valuation_date.isoformat() if holding.valuation_date else None,
                "unrealized_gain_loss": _num(holding.unrealized_gain_loss),
                "unrealized_gain_loss_percent": _num(holding.unrealized_gain_loss_percent),
                "percent_to_pms_portfolio": _num(holding.percent_to_pms_portfolio),
                "holding_period_months": holding.holding_period_months,
                "source_page": holding.source_page,
            })
    return {"rows": rows, "totals": _totals(rows, invested_key="investment_amount")}


def mutual_fund_holdings(reports: list[PortfolioReport]) -> dict:
    rows = []
    for report in reports:
        holding_periods_by_allocation = _sub_asset_holding_periods_by_allocation(report)
        for holding in report.mutual_fund_holdings:
            raw_values = _mutual_fund_values_from_raw_text(holding.raw_text)
            gain_loss_percent = holding.gain_loss_percent
            holding_period_months = holding.holding_period_months
            if gain_loss_percent is None:
                gain_loss_percent = raw_values.get("gain_loss_percent")
            if holding_period_months is None:
                holding_period_months = raw_values.get("holding_period_months")
            if holding_period_months is None:
                holding_period_months = holding_periods_by_allocation.get(_holding_allocation_key(holding))
            rows.append({
                "holder_name": report.holder_name,
                "scheme_name": holding.scheme_name,
                "asset_class": holding.asset_class,
                "asset_bucket": _asset_bucket(holding.asset_class),
                "sub_asset_class": holding.sub_asset_class,
                "investment_date": holding.investment_date.isoformat() if holding.investment_date else None,
                "units": _num(holding.units),
                "cost": _num(holding.cost),
                "current_value": _num(holding.market_value),
                "market_value": _num(holding.market_value),
                "unrealized_gain_loss": _num(holding.unrealized_gain_loss),
                "xirr_percent": _num(holding.xirr_percent),
                "dividend_paid": _num(holding.dividend_paid),
                "total_returns": _num(holding.total_returns),
                "total_gain": _num(holding.total_returns),
                "percent_to_mf_portfolio": _num(holding.percent_to_mf_portfolio),
                "purchase_price": _num(holding.purchase_price),
                "nav": _num(holding.nav),
                "gain_loss_percent": _num(gain_loss_percent),
                "holding_period_months": holding_period_months,
                "realized_gain_loss": _num(holding.realized_gain_loss),
                "folio_no": holding.folio_no,
                "source_page": holding.source_page,
            })
    return {"rows": rows, "totals": _totals(rows, invested_key="cost")}


def _sub_asset_holding_periods_by_allocation(report: PortfolioReport) -> dict:
    periods = {}
    current_asset_class = None
    top_level_names = {row.name for row in getattr(report, "asset_allocations", []) if hasattr(row, "name")}
    for row in report.sub_asset_allocations:
        if row.holding_period_months is None:
            continue
        asset_class = getattr(row, "asset_class", None)
        sub_asset_class = getattr(row, "sub_asset_class", None)
        if sub_asset_class is None and hasattr(row, "name"):
            if row.name in top_level_names or row.name == "Grand Total":
                current_asset_class = row.name
                continue
            asset_class = current_asset_class
            sub_asset_class = row.name
        key = _allocation_key(asset_class, sub_asset_class, row.current_value, row.invested_value)
        periods[key] = row.holding_period_months
    return periods


def _holding_allocation_key(holding) -> tuple:
    return _allocation_key(holding.asset_class, holding.sub_asset_class, holding.market_value, holding.cost)


def _allocation_key(asset_class: str | None, sub_asset_class: str | None, current_value, invested_value) -> tuple:
    return (
        asset_class,
        sub_asset_class,
        Decimal(str(current_value)) if current_value is not None else None,
        Decimal(str(invested_value)) if invested_value is not None else None,
    )


def _mutual_fund_values_from_raw_text(raw_text: str | None) -> dict:
    if not raw_text:
        return {}
    match = re.search(r"(?P<date>\d{4})(?P<rest>.*)$", raw_text)
    if not match:
        return {}
    return extract_mutual_fund_values(match.group("rest"))


def bond_analysis(reports: list[PortfolioReport]) -> dict:
    rows = []
    for report in reports:
        for holding in report.bond_holdings:
            rows.append({
                "holder_name": report.holder_name,
                "security_name": holding.security_name,
                "rating": holding.rating,
                "investment_date": holding.investment_date.isoformat() if holding.investment_date else None,
                "maturity_date": holding.maturity_date.isoformat() if holding.maturity_date else None,
                "cost": _num(holding.cost),
                "market_value": _num(holding.market_value),
                "interest_income": _num(holding.interest_income),
                "unrealized_gain_loss": _num(holding.unrealized_gain_loss),
                "unrealized_gain_loss_percent": _num(holding.unrealized_gain_loss_percent),
                "source_page": holding.source_page,
            })
    return {"rows": rows, "totals": _totals(rows, invested_key="cost")}


def report_sources(reports: list[PortfolioReport]) -> list[dict]:
    return [
        {
            "source_file_name": report.source_file_name,
            "holder_name": report.holder_name,
            "report_date": report.report_date.isoformat(),
        }
        for report in reports
    ]


def _empty_amount_row() -> dict:
    return {
        "current_value": Decimal("0"),
        "invested_value": Decimal("0"),
        "unrealized_gain_loss": Decimal("0"),
        "dividend_interest_paid": Decimal("0"),
        "total_gain": Decimal("0"),
        "realized_gain_loss": Decimal("0"),
    }


def _add_amounts(target: dict, row):
    asset_class = getattr(row, "asset_class", getattr(row, "name", None))
    target["asset_class"] = asset_class
    target["asset_bucket"] = _asset_bucket(asset_class)
    target["current_value"] += row.current_value or Decimal("0")
    target["invested_value"] += row.invested_value or Decimal("0")
    target["unrealized_gain_loss"] += row.unrealized_gain_loss or Decimal("0")
    target["dividend_interest_paid"] += row.dividend_interest_paid or Decimal("0")
    target["total_gain"] += row.total_gain or Decimal("0")
    target["realized_gain_loss"] += row.realized_gain_loss or Decimal("0")


def _with_recomputed_allocations(rows_or_mapping: Iterable[dict] | dict) -> list[dict]:
    rows = list(rows_or_mapping.values()) if isinstance(rows_or_mapping, dict) else list(rows_or_mapping)
    total_current_value = sum((row.get("current_value") or Decimal("0") for row in rows), Decimal("0"))
    output = []
    for row in rows:
        current_value = row.get("current_value") or Decimal("0")
        allocation = Decimal("0")
        if total_current_value:
            allocation = (current_value / total_current_value * Decimal("100")).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        output.append({
            **{key: value for key, value in row.items() if key not in {"current_value", "invested_value", "unrealized_gain_loss", "dividend_interest_paid", "total_gain", "realized_gain_loss"}},
            "current_value": _num(row.get("current_value")),
            "invested_value": _num(row.get("invested_value")),
            "unrealized_gain_loss": _num(row.get("unrealized_gain_loss")),
            "dividend_interest_paid": _num(row.get("dividend_interest_paid")),
            "total_gain": _num(row.get("total_gain")),
            "realized_gain_loss": _num(row.get("realized_gain_loss")),
            "current_allocation_percent": _num(allocation),
        })
    return output


def _allocation_row_to_dict(row, holder_name: str) -> dict:
    asset_bucket = _asset_bucket(row.asset_class)
    return {
        "holder_name": holder_name,
        "asset_class": row.asset_class,
        "asset_bucket": asset_bucket,
        "current_value": _num(row.current_value),
        "invested_value": _num(row.invested_value),
        "unrealized_gain_loss": _num(row.unrealized_gain_loss),
        "dividend_interest_paid": _num(row.dividend_interest_paid),
        "xirr_percent": _num(row.xirr_percent),
        "current_allocation_percent": _num(row.current_allocation_percent),
        "total_gain": _num(row.total_gain),
        "holding_period_months": row.holding_period_months,
        "realized_gain_loss": _num(row.realized_gain_loss),
        "source_page": row.source_page,
    }


def _sub_allocation_row_to_dict(row, holder_name: str) -> dict:
    data = _allocation_row_to_dict(row, holder_name)
    data["sub_asset_class"] = row.sub_asset_class
    return data


def _asset_bucket(asset_class: str | None) -> str | None:
    if not asset_class:
        return None
    normalized = asset_class.lower()
    if "equity" in normalized or "hybrid" in normalized:
        return "Equity"
    if any(term in normalized for term in ("debt", "fixed income", "cash", "bond", "liquid", "fd")):
        return "Debt"
    return asset_class


def _totals(rows: list[dict], invested_key: str = "invested_value") -> dict:
    return {
        "current_value": _num(sum((Decimal(str(row.get("current_value") or 0)) for row in rows), Decimal("0"))),
        "invested_value": _num(sum((Decimal(str(row.get(invested_key) or 0)) for row in rows), Decimal("0"))),
        "total_gain": _num(sum((Decimal(str(row.get("total_gain") or row.get("unrealized_gain_loss") or 0)) for row in rows), Decimal("0"))),
    }


def _num(value) -> float | None:
    if value is None:
        return None
    return float(value)
