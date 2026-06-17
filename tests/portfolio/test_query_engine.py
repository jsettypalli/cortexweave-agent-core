from cortexweave_core.rag.pipelines.portfolio.executor import execute_portfolio_plan
from cortexweave_core.rag.pipelines.portfolio.planner import (
    build_portfolio_query_plan,
    portfolio_schema_from_context,
)
from cortexweave_core.rag.pipelines.portfolio.query_engine import (
    _structured_fallback_answer,
)


def _execute(question: str, context: dict):
    plan = build_portfolio_query_plan(question, portfolio_schema_from_context(context))
    return plan, execute_portfolio_plan(plan, context)


def test_highest_and_lowest_xirr_route_to_mutual_fund_holdings():
    context = {
        "mutual_fund_holdings": {
            "rows": [
                {"scheme_name": "Low XIRR Fund", "xirr_percent": 2.5},
                {"scheme_name": "High XIRR Fund", "xirr_percent": 18.0},
            ]
        }
    }

    high_plan, high_result = _execute("Which fund has had the highest XIRR?", context)
    low_plan, low_result = _execute("Which fund has had the lowest XIRR?", context)

    assert high_plan.datasets == ["mutual_fund_holdings"]
    assert high_plan.sort == {"field": "xirr_percent", "direction": "desc"}
    assert (
        high_result["datasets"]["mutual_fund_holdings"]["rows"][0]["scheme_name"]
        == "High XIRR Fund"
    )
    assert low_plan.sort == {"field": "xirr_percent", "direction": "asc"}
    assert (
        low_result["datasets"]["mutual_fund_holdings"]["rows"][0]["scheme_name"]
        == "Low XIRR Fund"
    )


def test_holder_specific_longest_and_shortest_held_funds():
    context = {
        "mutual_fund_holdings": {
            "rows": [
                {
                    "holder_name": "SULEMANJI M ROOWALA",
                    "scheme_name": "Sulemanji Long Fund",
                    "holding_period_months": 120,
                },
                {
                    "holder_name": "SULEMANJI M ROOWALA",
                    "scheme_name": "Sulemanji Short Fund",
                    "holding_period_months": 12,
                },
                {
                    "holder_name": "Ms. ROOWALLA FATEMA SULEMANJI",
                    "scheme_name": "Fatema Short Fund",
                    "holding_period_months": 6,
                },
                {
                    "holder_name": "Ms. ROOWALLA FATEMA SULEMANJI",
                    "scheme_name": "Fatema Long Fund",
                    "holding_period_months": 80,
                },
            ]
        }
    }

    long_plan, long_result = _execute(
        "Which fund of Sulemanji Roowala has been held the longest?",
        context,
    )
    short_plan, short_result = _execute(
        "Which fund of Fatema Roowalla has been held the shortest?",
        context,
    )

    assert long_plan.filters == {"holder_name": "SULEMANJI M ROOWALA"}
    assert long_plan.sort == {"field": "holding_period_months", "direction": "desc"}
    assert (
        long_result["datasets"]["mutual_fund_holdings"]["rows"][0]["scheme_name"]
        == "Sulemanji Long Fund"
    )
    assert short_plan.filters == {"holder_name": "Ms. ROOWALLA FATEMA SULEMANJI"}
    assert short_plan.sort == {"field": "holding_period_months", "direction": "asc"}
    assert (
        short_result["datasets"]["mutual_fund_holdings"]["rows"][0]["scheme_name"]
        == "Fatema Short Fund"
    )


def test_equity_fund_highest_gain_for_holder_uses_total_gain_not_current_value():
    context = {
        "mutual_fund_holdings": {
            "rows": [
                {
                    "holder_name": "SULEMANJI M ROOWALA",
                    "scheme_name": "Large Low Gain Fund",
                    "asset_class": "Equity",
                    "current_value": 1000.0,
                    "total_gain": 10.0,
                },
                {
                    "holder_name": "SULEMANJI M ROOWALA",
                    "scheme_name": "Small High Gain Fund",
                    "asset_class": "Equity",
                    "current_value": 100.0,
                    "total_gain": 500.0,
                },
            ]
        }
    }

    plan, result = _execute(
        "Which Equity fund in Sulemanji Roowala's portfolio has had the highest gain?",
        context,
    )

    assert plan.filters == {
        "holder_name": "SULEMANJI M ROOWALA",
        "asset_class": "Equity",
    }
    assert plan.sort == {"field": "total_gain", "direction": "desc"}
    assert (
        result["datasets"]["mutual_fund_holdings"]["rows"][0]["scheme_name"]
        == "Small High Gain Fund"
    )


def test_holder_returns_can_rank_highest_total_portfolio_gain():
    context = {
        "holder_returns": {
            "rows": [
                {"holder_name": "Holder A", "total_gain": 100.0},
                {"holder_name": "Holder B", "total_gain": 250.0},
            ]
        }
    }

    plan, result = _execute("Which holder has the highest total portfolio gain?", context)

    assert plan.datasets == ["holder_returns"]
    assert plan.sort == {"field": "total_gain", "direction": "desc"}
    assert result["datasets"]["holder_returns"]["rows"][0]["holder_name"] == "Holder B"


def test_asset_allocation_for_holder_uses_holder_rows():
    context = {
        "asset_allocations": {
            "rows": [{"asset_class": "Equity", "current_value": 300.0}],
            "holder_rows": [
                {
                    "holder_name": "SULEMANJI M ROOWALA",
                    "asset_class": "Equity",
                    "current_value": 100.0,
                },
                {
                    "holder_name": "SULEMANJI M ROOWALA",
                    "asset_class": "Debt/Fixed Income",
                    "current_value": 50.0,
                },
            ],
        }
    }

    plan, result = _execute("What is the asset allocation for Sulemanji Roowala?", context)

    rows = result["datasets"]["asset_allocations"]["rows"]
    assert plan.filters == {"holder_name": "SULEMANJI M ROOWALA"}
    assert len(rows) == 2
    assert {row["asset_class"] for row in rows} == {"Equity", "Debt/Fixed Income"}


def test_debt_fund_lowest_return_and_highest_gain_percentage():
    context = {
        "mutual_fund_holdings": {
            "rows": [
                {
                    "scheme_name": "Debt Low Return",
                    "asset_bucket": "Debt",
                    "asset_class": "Debt/Fixed Income",
                    "xirr_percent": 3.0,
                    "gain_loss_percent": 4.0,
                },
                {
                    "scheme_name": "Debt High Return",
                    "asset_bucket": "Debt",
                    "asset_class": "Debt/Fixed Income",
                    "xirr_percent": 7.0,
                    "gain_loss_percent": 9.0,
                },
                {
                    "scheme_name": "Equity High Gain Percent",
                    "asset_bucket": "Equity",
                    "asset_class": "Equity",
                    "gain_loss_percent": 25.0,
                },
            ]
        }
    }

    low_plan, low_result = _execute("Which debt fund has the lowest return?", context)
    gain_plan, gain_result = _execute(
        "Which fund has the highest gain percentage?",
        context,
    )

    assert low_plan.filters == {"asset_bucket": "Debt"}
    assert low_plan.sort == {"field": "xirr_percent", "direction": "asc"}
    assert (
        low_result["datasets"]["mutual_fund_holdings"]["rows"][0]["scheme_name"]
        == "Debt Low Return"
    )
    assert gain_plan.sort == {"field": "gain_loss_percent", "direction": "desc"}
    assert (
        gain_result["datasets"]["mutual_fund_holdings"]["rows"][0]["scheme_name"]
        == "Equity High Gain Percent"
    )


def test_structured_fallback_answer_uses_sorted_row_metric():
    answer = _structured_fallback_answer(
        "Which fund has had the lowest XIRR?",
        {
            "plan": {"sort": {"field": "xirr_percent", "direction": "asc"}},
            "datasets": {
                "mutual_fund_holdings": {
                    "rows": [{"scheme_name": "Low XIRR Fund", "xirr_percent": 2.5}]
                }
            },
        },
    )

    assert answer == "The lowest matching result is Low XIRR Fund with xirr_percent 2.5."


def test_structured_fallback_answer_uses_holder_name_for_holder_returns():
    answer = _structured_fallback_answer(
        "Which holder has the highest total portfolio gain?",
        {
            "plan": {"sort": {"field": "total_gain", "direction": "desc"}},
            "datasets": {
                "holder_returns": {
                    "rows": [
                        {
                            "holder_name": "SULEMANJI M ROOWALA",
                            "asset_class": "Grand Total",
                            "total_gain": 248155010.0,
                        }
                    ]
                }
            },
        },
    )

    assert (
        answer
        == "The highest matching result is SULEMANJI M ROOWALA with total_gain 248155010.0."
    )


def test_nav_question_routes_to_exact_mutual_fund_holding():
    context = {
        "sub_asset_allocations": {
            "rows": [
                {
                    "holder_name": "SULEMANJI M ROOWALA",
                    "sub_asset_class": "Flexi Cap",
                    "current_value": 8910823.0,
                }
            ]
        },
        "mutual_fund_holdings": {
            "rows": [
                {
                    "holder_name": "SULEMANJI M ROOWALA",
                    "scheme_name": "Canara Robeco Flexi Cap Reg-G",
                    "sub_asset_class": "Flexi Cap",
                    "nav": 301.28,
                }
            ]
        },
    }

    plan, result = _execute(
        "What is the NAV for Canara Robeco Flexi Cap Reg-G for Sulemanji?",
        context,
    )

    assert plan.datasets == ["mutual_fund_holdings"]
    assert plan.filters == {
        "holder_name": "SULEMANJI M ROOWALA",
        "sub_asset_class": "Flexi Cap",
        "scheme_name": "Canara Robeco Flexi Cap Reg-G",
    }
    assert plan.metrics == ["nav"]
    assert result["datasets"]["mutual_fund_holdings"]["rows"][0]["nav"] == 301.28


def test_structured_fallback_answer_reports_requested_nav_metric():
    answer = _structured_fallback_answer(
        "What is the NAV?",
        {
            "plan": {"metrics": ["nav"]},
            "datasets": {
                "mutual_fund_holdings": {
                    "rows": [
                        {
                            "scheme_name": "Canara Robeco Flexi Cap Reg-G",
                            "nav": 301.28,
                        }
                    ]
                }
            },
        },
    )

    assert answer == "nav for Canara Robeco Flexi Cap Reg-G is 301.28."


def test_purchase_price_question_routes_to_exact_mutual_fund_holding():
    context = {
        "sub_asset_allocations": {
            "rows": [
                {
                    "holder_name": "SULEMANJI M ROOWALA",
                    "sub_asset_class": "Flexi Cap",
                    "current_value": 8910823.0,
                }
            ]
        },
        "mutual_fund_holdings": {
            "rows": [
                {
                    "holder_name": "SULEMANJI M ROOWALA",
                    "scheme_name": "Canara Robeco Flexi Cap Reg-G",
                    "sub_asset_class": "Flexi Cap",
                    "purchase_price": 148.369,
                }
            ]
        },
    }

    plan, result = _execute(
        "What is the purchase price for Canara Robeco Flexi Cap Reg-G for Sulemanji?",
        context,
    )

    assert plan.datasets == ["mutual_fund_holdings"]
    assert plan.filters["scheme_name"] == "Canara Robeco Flexi Cap Reg-G"
    assert plan.metrics == ["purchase_price"]
    assert (
        result["datasets"]["mutual_fund_holdings"]["rows"][0]["purchase_price"]
        == 148.369
    )
