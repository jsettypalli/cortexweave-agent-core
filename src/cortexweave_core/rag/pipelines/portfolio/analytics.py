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


def fund_resolution_status(reports: list[PortfolioReport]) -> dict:
    rows = []
    for report in reports:
        for resolution in getattr(report, "fund_resolutions", []) or []:
            rows.append({
                "holder_name": report.holder_name,
                "source_file_name": report.source_file_name,
                "resolution_id": resolution.id,
                "mutual_fund_holding_id": resolution.mutual_fund_holding_id,
                "scheme_name": resolution.raw_scheme_name,
                "statement_category": resolution.statement_category,
                "scheme_code": resolution.scheme_code,
                "matched_name": resolution.matched_name,
                "status": resolution.status,
                "confidence": resolution.confidence,
                "score": _num(resolution.score),
                "lead": _num(resolution.lead),
                "candidates": resolution.alternatives_json or [],
                "error_message": resolution.error_message,
            })
    return {"rows": rows, "totals": {"total_funds": len(rows)}}


def fund_stock_holdings(reports: list[PortfolioReport]) -> dict:
    rows = []
    for report in reports:
        for resolution in getattr(report, "fund_resolutions", []) or []:
            holding = resolution.mutual_fund_holding
            for exposure in resolution.security_exposures:
                rows.append({
                    "holder_name": report.holder_name,
                    "scheme_name": holding.scheme_name if holding else resolution.raw_scheme_name,
                    "matched_name": resolution.matched_name,
                    "scheme_code": resolution.scheme_code,
                    "asset_class": holding.asset_class if holding else None,
                    "asset_bucket": _asset_bucket(holding.asset_class if holding else None),
                    "sub_asset_class": holding.sub_asset_class if holding else None,
                    "stock_name": exposure.stock_name,
                    "sector": exposure.sector,
                    "instrument": exposure.instrument,
                    "nature": exposure.nature,
                    "pct_of_fund_assets": _num(exposure.pct_of_fund_assets),
                    "fund_current_value": _num(exposure.fund_current_value),
                    "portfolio_weighted_pct": _num(exposure.portfolio_weighted_pct),
                    "weighted_market_value": _num(exposure.weighted_market_value),
                    "as_of_date": exposure.as_of_date,
                })
    return {"rows": rows, "totals": _totals(rows, invested_key="weighted_market_value")}


def stock_overlap(reports: list[PortfolioReport], min_fund_count: int = 2) -> dict:
    rows = fund_stock_holdings(reports)["rows"]
    grouped: dict[str, dict] = {}
    for row in rows:
        stock = row.get("stock_name")
        if not stock:
            continue
        target = grouped.setdefault(str(stock), {
            "stock_name": stock,
            "sector": row.get("sector"),
            "nature": row.get("nature"),
            "funds": [],
            "aggregate_portfolio_exposure_percent": 0.0,
            "aggregate_weighted_market_value": 0.0,
            "max_pct_in_single_fund": 0.0,
        })
        fund_label = row.get("matched_name") or row.get("scheme_name")
        if fund_label and fund_label not in target["funds"]:
            target["funds"].append(fund_label)
        target["aggregate_portfolio_exposure_percent"] += float(row.get("portfolio_weighted_pct") or 0)
        target["aggregate_weighted_market_value"] += float(row.get("weighted_market_value") or 0)
        target["max_pct_in_single_fund"] = max(
            target["max_pct_in_single_fund"],
            float(row.get("pct_of_fund_assets") or 0),
        )
    output = []
    for row in grouped.values():
        row["fund_count"] = len(row["funds"])
        if row["fund_count"] >= min_fund_count:
            row["aggregate_portfolio_exposure_percent"] = round(row["aggregate_portfolio_exposure_percent"], 4)
            row["aggregate_weighted_market_value"] = round(row["aggregate_weighted_market_value"], 2)
            output.append(row)
    output.sort(key=lambda row: row["aggregate_portfolio_exposure_percent"], reverse=True)
    return {"rows": output, "totals": {"overlap_count": len(output)}}


def sector_overlap(reports: list[PortfolioReport]) -> dict:
    rows = fund_stock_holdings(reports)["rows"]
    grouped: dict[str, dict] = {}
    for row in rows:
        sector = row.get("sector") or "Unclassified"
        target = grouped.setdefault(str(sector), {
            "sector": sector,
            "funds": {},
            "aggregate_portfolio_exposure_percent": 0.0,
            "aggregate_weighted_market_value": 0.0,
        })
        fund_label = row.get("matched_name") or row.get("scheme_name")
        if fund_label:
            target["funds"][fund_label] = target["funds"].get(fund_label, 0.0) + float(row.get("portfolio_weighted_pct") or 0)
        target["aggregate_portfolio_exposure_percent"] += float(row.get("portfolio_weighted_pct") or 0)
        target["aggregate_weighted_market_value"] += float(row.get("weighted_market_value") or 0)
    output = []
    for row in grouped.values():
        top = sorted(row["funds"].items(), key=lambda item: item[1], reverse=True)[:5]
        output.append({
            "sector": row["sector"],
            "fund_count": len(row["funds"]),
            "aggregate_portfolio_exposure_percent": round(row["aggregate_portfolio_exposure_percent"], 4),
            "aggregate_weighted_market_value": round(row["aggregate_weighted_market_value"], 2),
            "top_contributing_funds": [name for name, _ in top],
        })
    output.sort(key=lambda row: row["aggregate_portfolio_exposure_percent"], reverse=True)
    return {"rows": output, "totals": {"sector_count": len(output)}}


def fund_overlap_matrix(reports: list[PortfolioReport]) -> dict:
    by_fund: dict[str, set[str]] = defaultdict(set)
    for row in fund_stock_holdings(reports)["rows"]:
        fund = row.get("matched_name") or row.get("scheme_name")
        stock = row.get("stock_name")
        if fund and stock:
            by_fund[str(fund)].add(str(stock))
    funds = sorted(by_fund)
    rows = []
    for i, fund_a in enumerate(funds):
        for fund_b in funds[i + 1:]:
            shared = sorted(by_fund[fund_a] & by_fund[fund_b])
            if not shared:
                continue
            rows.append({
                "fund_a": fund_a,
                "fund_b": fund_b,
                "shared_stocks": len(shared),
                "stocks": shared,
            })
    rows.sort(key=lambda row: row["shared_stocks"], reverse=True)
    return {"rows": rows, "totals": {"pairs_with_overlap": len(rows)}}


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
