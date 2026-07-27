from cortexweave_core.rag.pipelines.portfolio.executor import execute_portfolio_plan
from cortexweave_core.rag.pipelines.portfolio import analytics
from cortexweave_core.rag.pipelines.portfolio.planner import (
    build_portfolio_query_plan,
    portfolio_schema_from_context,
)
from cortexweave_core.rag.pipelines.portfolio.query_engine import (
    _asset_class_segregation_answer,
    _enrichment_answer,
    _requested_holder_names,
    _sub_asset_parent_allocation_answer,
    _structured_fallback_answer,
)


def _execute(question: str, context: dict):
    plan = build_portfolio_query_plan(question, portfolio_schema_from_context(context))
    return plan, execute_portfolio_plan(plan, context)


def test_overlap_exposures_use_weighted_value_and_scope_denominator():
    exposures = [
        {
            "holder_name": "Small Holder",
            "matched_name": "Fund A",
            "stock_name": "Shared Stock",
            "sector": "Financial",
            "portfolio_weighted_pct": 50,
            "weighted_market_value": 50,
            "fund_current_value": 100,
            "pct_of_fund_assets": 50,
        },
        {
            "holder_name": "Large Holder",
            "matched_name": "Fund B",
            "stock_name": "Shared Stock",
            "sector": "Financial",
            "portfolio_weighted_pct": 10,
            "weighted_market_value": 900,
            "fund_current_value": 9000,
            "pct_of_fund_assets": 10,
        },
    ]

    sector = analytics.sector_overlap_rows(exposures, 10000)[0]
    stock = analytics.stock_overlap_rows(exposures, 10000)[0]
    fund_pair = analytics.fund_overlap_rows(exposures, 10000)[0]

    assert sector["aggregate_portfolio_exposure_percent"] == 9.5
    assert sector["top_contributing_funds"] == ["Fund B", "Fund A"]
    assert stock["aggregate_portfolio_exposure_percent"] == 9.5
    assert fund_pair["shared_portfolio_exposure_percent"] == 9.5
    assert analytics.exposure_metadata(10000, exposures)["coverage_percent"] == 9.5


def test_stock_overlap_understands_more_than_one_fund_holder_and_top_limit():
    fatema = "Ms. ROOWALLA FATEMA SULEMANJI"
    rows = []
    for idx in range(11):
        for fund in ("Fund A", "Fund B"):
            rows.append({
                "holder_name": fatema,
                "stock_name": f"Stock {idx:02d}",
                "sector": "Financial",
                "nature": "EQUITY",
                "matched_name": fund,
                "portfolio_weighted_pct": 20 - idx,
                "weighted_market_value": 1000 - idx,
                "pct_of_fund_assets": 5 + idx,
            })
    rows.extend([
        {
            "holder_name": fatema,
            "stock_name": "Repo",
            "nature": "CASH",
            "matched_name": "Cash Fund A",
            "portfolio_weighted_pct": 500,
            "weighted_market_value": 5000,
            "pct_of_fund_assets": 50,
        },
        {
            "holder_name": fatema,
            "stock_name": "Repo",
            "nature": "CASH",
            "matched_name": "Cash Fund B",
            "portfolio_weighted_pct": 500,
            "weighted_market_value": 5000,
            "pct_of_fund_assets": 50,
        },
    ])

    result = _enrichment_answer(
        "Top 10 stocks held by more than one mutual fund for Fatema",
        {
            "fund_resolution_status": {"rows": []},
            "fund_stock_holdings": {"rows": rows},
            "holder_returns": {"rows": [{"holder_name": fatema}, {"holder_name": "Other Holder"}]},
        },
    )

    dataset = result["data"]["datasets"]["stock_overlap"]
    table_rows = result["tables"][0]["rows"]
    assert result["intent"] == "stock_overlap"
    assert dataset["holder_names"] == [fatema]
    assert dataset["matched_rows"] == 10
    assert dataset["total_rows"] == 11
    assert len(table_rows) == 10
    assert table_rows[0]["stock_name"] == "Stock 00"
    assert "Repo" not in {row["stock_name"] for row in table_rows}


def test_stock_overlap_keeps_default_payload_capped_for_llm_usage():
    rows = []
    for idx in range(60):
        for fund in ("Fund A", "Fund B"):
            rows.append({
                "stock_name": f"Stock {idx:02d}",
                "nature": "EQUITY",
                "matched_name": fund,
                "portfolio_weighted_pct": 100 - idx,
                "weighted_market_value": 1000 - idx,
                "pct_of_fund_assets": 5,
            })

    result = _enrichment_answer(
        "Show me overlapping stock holdings across my mutual funds",
        {
            "fund_resolution_status": {"rows": []},
            "fund_stock_holdings": {"rows": rows},
            "holder_returns": {"rows": []},
        },
        knowledge_base_name="kb",
        family_id="family",
    )

    dataset = result["data"]["datasets"]["stock_overlap"]
    table = result["tables"][0]
    assert dataset["matched_rows"] == 10
    assert dataset["total_rows"] == 60
    assert len(dataset["rows"]) == 10
    assert len(table["rows"]) == 10
    assert table["query_ref"] is not None

    top_result = _enrichment_answer(
        "Show me top 25 overlapping stock holdings across my mutual funds",
        {
            "fund_resolution_status": {"rows": []},
            "fund_stock_holdings": {"rows": rows},
            "holder_returns": {"rows": []},
        },
        knowledge_base_name="kb",
        family_id="family",
    )

    top_dataset = top_result["data"]["datasets"]["stock_overlap"]
    top_table = top_result["tables"][0]
    assert top_dataset["matched_rows"] == 25
    assert len(top_dataset["rows"]) == 25
    assert len(top_table["rows"]) == 25
    assert top_table["query_ref"] is not None
    assert top_table["rows_complete"] is True


def test_security_overlap_can_include_non_equity_holdings():
    fatema = "Ms. ROOWALLA FATEMA SULEMANJI"
    rows = [
        {
            "holder_name": fatema,
            "stock_name": "Repo",
            "nature": "CASH",
            "matched_name": "Cash Fund A",
            "portfolio_weighted_pct": 500,
            "weighted_market_value": 5000,
            "pct_of_fund_assets": 50,
        },
        {
            "holder_name": fatema,
            "stock_name": "Repo",
            "nature": "CASH",
            "matched_name": "Cash Fund B",
            "portfolio_weighted_pct": 500,
            "weighted_market_value": 5000,
            "pct_of_fund_assets": 50,
        },
    ]

    result = _enrichment_answer(
        "Show me top overlapping securities across my mutual funds",
        {
            "fund_resolution_status": {"rows": []},
            "fund_stock_holdings": {"rows": rows},
            "holder_returns": {"rows": [{"holder_name": fatema}]},
        },
    )

    assert result["data"]["datasets"]["stock_overlap"]["nature"] is None
    assert result["tables"][0]["rows"][0]["stock_name"] == "Repo"


def test_debt_holding_overlap_filters_to_duplicated_debt_securities():
    rows = [
        {
            "stock_name": "Government Bond 2035",
            "sector": "Sovereign",
            "nature": "DEBT",
            "matched_name": fund,
            "portfolio_weighted_pct": 2,
            "weighted_market_value": 2000,
            "pct_of_fund_assets": 5,
        }
        for fund in ("Debt Fund A", "Debt Fund B")
    ]
    rows.extend([
        {
            "stock_name": "Equity Stock",
            "sector": "Financial",
            "nature": "EQUITY",
            "matched_name": fund,
            "portfolio_weighted_pct": 20,
            "weighted_market_value": 20000,
            "pct_of_fund_assets": 10,
        }
        for fund in ("Equity Fund A", "Equity Fund B")
    ])

    result = _enrichment_answer(
        "Show me top 10 overlapping debt holdings across my mutual funds.",
        {
            "fund_resolution_status": {"rows": []},
            "fund_stock_holdings": {"rows": rows},
            "holder_returns": {"rows": []},
        },
    )

    dataset = result["data"]["datasets"]["stock_overlap"]
    table = result["tables"][0]
    assert result["intent"] == "stock_overlap"
    assert dataset["nature"] == "DEBT"
    assert dataset["total_rows"] == 1
    assert dataset["rows"][0]["stock_name"] == "Government Bond 2035"
    assert table["title"] == "Duplicated Debt Holding Exposure"
    assert table["columns"][0]["label"] == "Debt Holding"


def test_security_exposure_routes_combined_exposure_without_overlap_requirement():
    rows = [
        {
            "stock_name": "Single Fund Bond",
            "nature": "DEBT",
            "matched_name": "Debt Fund A",
            "portfolio_weighted_pct": 25,
            "weighted_market_value": 2500,
            "pct_of_fund_assets": 25,
        },
        {
            "stock_name": "Shared Stock",
            "nature": "EQUITY",
            "matched_name": "Equity Fund A",
            "portfolio_weighted_pct": 10,
            "weighted_market_value": 1000,
            "pct_of_fund_assets": 10,
        },
        {
            "stock_name": "Shared Stock",
            "nature": "EQUITY",
            "matched_name": "Equity Fund B",
            "portfolio_weighted_pct": 9,
            "weighted_market_value": 900,
            "pct_of_fund_assets": 9,
        },
    ]

    result = _enrichment_answer(
        "Which securities have the highest combined exposure across my mutual fund holdings?",
        {
            "fund_resolution_status": {"rows": []},
            "fund_stock_holdings": {"rows": rows},
            "holder_returns": {"rows": []},
        },
    )

    dataset = result["data"]["datasets"]["stock_overlap"]
    table = result["tables"][0]
    assert result["intent"] == "security_exposure"
    assert dataset["matched_rows"] == 2
    assert dataset["nature"] is None
    assert table["title"] == "Underlying Security Exposure"
    assert table["rows"][0]["stock_name"] == "Single Fund Bond"
    assert table["rows"][0]["fund_count"] == 1


def test_fund_overlap_respects_top_limit():
    rows = [
        {
            "fund_a": f"Fund {idx:02d}A",
            "fund_b": f"Fund {idx:02d}B",
            "shared_stocks": 100 - idx,
            "shared_portfolio_exposure_percent": 100 - idx,
            "stocks": [
                {"name": f"Stock {stock_idx:02d}", "combined_portfolio_exposure_percent": 20 - stock_idx}
                for stock_idx in range(12)
            ],
        }
        for idx in range(12)
    ]

    result = _enrichment_answer(
        "Show top 10 fund overlap by underlying stocks, sorted by highest portfolio exposure.",
        {
            "fund_resolution_status": {"rows": []},
            "fund_overlap_matrix": {"rows": rows},
        },
    )

    dataset = result["data"]["datasets"]["fund_overlap_matrix"]
    table = result["tables"][0]
    assert result["intent"] == "fund_overlap_matrix"
    assert dataset["matched_rows"] == 10
    assert dataset["total_rows"] == 12
    assert len(dataset["rows"][0]["stocks"]) == 10
    assert len(table["rows"]) == 10
    assert len(table["rows"][0]["stocks"]) == 10
    assert table["warnings"] == [
        "Showing top 10 rows by exposure out of 12 total matches.",
        "Shared stock lists are capped to top 10 by portfolio exposure in this view.",
    ]
    assert table["query_ref"] is None


def test_fund_overlap_by_underlying_stocks_excludes_non_equity_instruments():
    exposure_rows = [
        {
            "matched_name": fund,
            "stock_name": "Shared Equity",
            "nature": "EQUITY",
            "weighted_market_value": value,
            "fund_current_value": 1000,
            "pct_of_fund_assets": value / 10,
        }
        for fund, value in (("Fund A", 100), ("Fund B", 200))
    ] + [
        {
            "matched_name": fund,
            "stock_name": "Cash Margin",
            "nature": "CASH",
            "weighted_market_value": 5000,
            "fund_current_value": 1000,
            "pct_of_fund_assets": 500,
        }
        for fund in ("Fund A", "Fund B", "Fund C")
    ]

    result = _enrichment_answer(
        "Show top 10 fund overlap by underlying stocks, sorted by highest portfolio exposure.",
        {
            "fund_resolution_status": {"rows": []},
            "fund_stock_holdings": {"rows": exposure_rows},
            "fund_overlap_matrix": {"rows": [{
                "fund_a": "Cached Mixed Fund A",
                "fund_b": "Cached Mixed Fund B",
                "shared_stocks": 1,
                "shared_portfolio_exposure_percent": 999,
                "stocks": [{"name": "Cash Margin"}],
            }]},
        },
    )

    table_rows = result["tables"][0]["rows"]
    assert result["data"]["datasets"]["fund_overlap_matrix"]["total_rows"] == 1
    assert len(table_rows) == 1
    assert {table_rows[0]["fund_a"], table_rows[0]["fund_b"]} == {"Fund A", "Fund B"}
    assert table_rows[0]["shared_stocks"] == 1
    assert [stock["name"] for stock in table_rows[0]["stocks"]] == ["Shared Equity"]


def test_fund_overlap_by_underlying_debt_filters_instruments_and_fund_scope():
    exposure_rows = [
        {
            "matched_name": "Equity Fund A",
            "asset_bucket": "Equity",
            "stock_name": "Shared Bond 1",
            "nature": "DEBT",
            "weighted_market_value": 100,
        },
        {
            "matched_name": "Debt Fund B",
            "asset_bucket": "Debt",
            "stock_name": "Shared Bond 1",
            "nature": "DEBT",
            "weighted_market_value": 200,
        },
        {
            "matched_name": "Debt Fund C",
            "asset_bucket": "Debt",
            "stock_name": "Shared Bond 2",
            "nature": "DEBT",
            "weighted_market_value": 300,
        },
        {
            "matched_name": "Debt Fund D",
            "asset_bucket": "Debt",
            "stock_name": "Shared Bond 2",
            "nature": "DEBT",
            "weighted_market_value": 400,
        },
        {
            "matched_name": "Equity Fund A",
            "asset_bucket": "Equity",
            "stock_name": "Shared Equity",
            "nature": "EQUITY",
            "weighted_market_value": 5000,
        },
        {
            "matched_name": "Debt Fund B",
            "asset_bucket": "Debt",
            "stock_name": "Shared Equity",
            "nature": "EQUITY",
            "weighted_market_value": 5000,
        },
    ]
    context = {
        "fund_resolution_status": {"rows": []},
        "fund_stock_holdings": {"rows": exposure_rows},
        "fund_overlap_matrix": {"rows": []},
    }

    result = _enrichment_answer(
        "Show top 10 fund overlap by underlying debt.",
        context,
    )

    dataset = result["data"]["datasets"]["fund_overlap_matrix"]
    table = result["tables"][0]
    assert dataset["total_rows"] == 2
    assert dataset["fund_asset_bucket"] is None
    assert dataset["instrument_nature"] == "DEBT"
    assert result["answer"] == "Found 2 fund pair(s) with shared underlying debt securities."
    assert [column["label"] for column in table["columns"]][2:6] == [
        "Shared Debt Securities",
        "Shared Exposure",
        "Shared Value",
        "Debt Securities",
    ]
    assert all(
        stock["name"].startswith("Shared Bond")
        for row in table["rows"]
        for stock in row["stocks"]
    )
    assert table["warnings"] == [
        "Shared debt security lists are capped to top 10 by portfolio exposure in this view.",
    ]

    debt_funds_result = _enrichment_answer(
        "Compare overlap between my debt mutual funds.",
        context,
    )
    debt_dataset = debt_funds_result["data"]["datasets"]["fund_overlap_matrix"]
    assert debt_dataset["total_rows"] == 1
    assert debt_dataset["fund_asset_bucket"] == "Debt"
    assert debt_dataset["instrument_nature"] == "DEBT"
    assert {
        debt_funds_result["tables"][0]["rows"][0]["fund_a"],
        debt_funds_result["tables"][0]["rows"][0]["fund_b"],
    } == {"Debt Fund C", "Debt Fund D"}

    natural_wording_result = _enrichment_answer(
        "Show me top 10 Debt overlaps in my Debt/Fixed Income funds",
        context,
    )
    natural_dataset = natural_wording_result["data"]["datasets"]["fund_overlap_matrix"]
    assert natural_wording_result["intent"] == "fund_overlap_matrix"
    assert natural_dataset["total_rows"] == 1
    assert natural_dataset["fund_asset_bucket"] == "Debt"
    assert natural_dataset["instrument_nature"] == "DEBT"

    plural_wording_result = _enrichment_answer(
        "Show me top 10 funds overlaps by underlying debt",
        context,
    )
    plural_dataset = plural_wording_result["data"]["datasets"]["fund_overlap_matrix"]
    assert plural_wording_result["intent"] == "fund_overlap_matrix"
    assert plural_dataset["total_rows"] == 2
    assert plural_dataset["fund_asset_bucket"] is None
    assert plural_dataset["instrument_nature"] == "DEBT"


def test_fund_overlap_filters_to_requested_holder():
    fatema = "Ms. ROOWALLA FATEMA SULEMANJI"
    other = "Other Holder"
    exposure_rows = [
        {
            "holder_name": fatema,
            "matched_name": fund,
            "stock_name": "Fatema Stock",
            "nature": "EQUITY",
            "portfolio_weighted_pct": exposure,
            "weighted_market_value": exposure * 100,
            "pct_of_fund_assets": exposure * 2,
        }
        for fund, exposure in (("Fatema Fund A", 2), ("Fatema Fund B", 3))
    ] + [
        {
            "holder_name": other,
            "matched_name": fund,
            "stock_name": "Other Stock",
            "nature": "EQUITY",
            "portfolio_weighted_pct": 100,
            "weighted_market_value": 10000,
            "pct_of_fund_assets": 50,
        }
        for fund in ("Other Fund A", "Other Fund B")
    ]

    result = _enrichment_answer(
        "Show top 10 fund overlap by underlying stocks, sorted by highest portfolio exposure for Fatema?",
        {
            "fund_resolution_status": {"rows": []},
            "fund_stock_holdings": {"rows": exposure_rows},
            "holder_returns": {"rows": [
                {"holder_name": fatema},
                {"holder_name": other},
            ]},
            "fund_overlap_matrix": {"rows": [{
                "fund_a": "Other Fund A",
                "fund_b": "Other Fund B",
                "shared_portfolio_exposure_percent": 200,
                "stocks": [{"name": "Other Stock"}],
            }]},
        },
    )

    dataset = result["data"]["datasets"]["fund_overlap_matrix"]
    assert dataset["holder_names"] == [fatema]
    assert dataset["total_rows"] == 1
    assert result["tables"][0]["rows"][0]["fund_a"] == "Fatema Fund A"
    assert result["tables"][0]["rows"][0]["fund_b"] == "Fatema Fund B"
    assert fatema in result["answer"]


def test_requested_holder_name_prefers_sulemanji_over_shared_name_token():
    sulemanji = "SULEMANJI M ROOWALA"
    fatema = "Ms. ROOWALLA FATEMA SULEMANJI"
    context = {
        "holder_returns": {"rows": [
            {"holder_name": sulemanji},
            {"holder_name": fatema},
        ]},
    }

    assert _requested_holder_names(context, "fund overlap for sulemanji") == [sulemanji]
    assert _requested_holder_names(context, "fund overlap for fatema") == [fatema]
    assert _requested_holder_names(
        context,
        "large cap as part of total assets and equity asset for fatema roowalla",
    ) == [fatema]


def test_fund_overlap_understands_between_equity_mutual_funds():
    rows = [
        {"scheme_name": "Equity Fund A", "asset_bucket": "Equity", "stock_name": "Stock 1", "nature": "EQUITY"},
        {"scheme_name": "Equity Fund A", "asset_bucket": "Equity", "stock_name": "Stock 2", "nature": "EQUITY"},
        {"scheme_name": "Equity Fund B", "asset_bucket": "Equity", "stock_name": "Stock 1", "nature": "EQUITY"},
        {"scheme_name": "Equity Fund C", "asset_bucket": "Equity", "stock_name": "Stock 2", "nature": "EQUITY"},
        {"scheme_name": "Equity Fund B", "asset_bucket": "Equity", "stock_name": "Cash Margin", "nature": "CASH"},
        {"scheme_name": "Equity Fund C", "asset_bucket": "Equity", "stock_name": "Cash Margin", "nature": "CASH"},
        {"scheme_name": "Debt Fund A", "asset_bucket": "Debt", "stock_name": "Stock 1", "nature": "EQUITY"},
        {"scheme_name": "Debt Fund A", "asset_bucket": "Debt", "stock_name": "Stock 2", "nature": "EQUITY"},
    ]

    result = _enrichment_answer(
        "Compare top 10 overlap between my equity mutual funds.",
        {
            "fund_resolution_status": {"rows": []},
            "fund_stock_holdings": {"rows": rows},
            "fund_overlap_matrix": {"rows": []},
        },
    )

    table = result["tables"][0]
    assert result["intent"] == "fund_overlap_matrix"
    assert result["answer"] == "Found 2 equity mutual fund pair(s) with shared underlying stocks."
    assert len(table["rows"]) == 2
    assert table["query_ref"] is None
    assert all("Debt Fund A" not in {row["fund_a"], row["fund_b"]} for row in table["rows"])
    assert all(
        stock["name"] != "Cash Margin"
        for row in table["rows"]
        for stock in row["stocks"]
    )

    all_holdings_result = _enrichment_answer(
        "Compare fund overlap across all holdings between my equity mutual funds.",
        {
            "fund_resolution_status": {"rows": []},
            "fund_stock_holdings": {"rows": rows},
            "fund_overlap_matrix": {"rows": []},
        },
    )

    assert all_holdings_result["data"]["datasets"]["fund_overlap_matrix"]["total_rows"] == 3


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


def test_highest_xirr_with_holding_period_range_ranks_by_holder():
    context = {
        "mutual_fund_holdings": {
            "rows": [
                {
                    "holder_name": "Holder One",
                    "scheme_name": "One Best In Range",
                    "asset_bucket": "Equity",
                    "asset_class": "Equity",
                    "xirr_percent": 13.8,
                    "holding_period_months": 77,
                },
                {
                    "holder_name": "Holder One",
                    "scheme_name": "One Excluded Too Long",
                    "asset_bucket": "Equity",
                    "asset_class": "Equity",
                    "xirr_percent": 30.0,
                    "holding_period_months": 120,
                },
                {
                    "holder_name": "Holder One",
                    "scheme_name": "One Lower In Range",
                    "asset_bucket": "Equity",
                    "asset_class": "Equity",
                    "xirr_percent": 12.5,
                    "holding_period_months": 43,
                },
                {
                    "holder_name": "Holder Two",
                    "scheme_name": "Two Best In Range",
                    "asset_bucket": "Equity",
                    "asset_class": "Equity",
                    "xirr_percent": 10.95,
                    "holding_period_months": 28,
                },
                {
                    "holder_name": "Holder Two",
                    "scheme_name": "Two Excluded Too Short",
                    "asset_bucket": "Equity",
                    "asset_class": "Equity",
                    "xirr_percent": 25.0,
                    "holding_period_months": 10,
                },
            ]
        }
    }

    question = (
        "Which fund has had the highest XIRR return in Equity which is atleast "
        "27 months and above but less than 100 months?"
    )
    plan, result = _execute(question, context)
    answer = _structured_fallback_answer(question, result)

    rows = result["datasets"]["mutual_fund_holdings"]["rows"]
    assert plan.datasets == ["mutual_fund_holdings"]
    assert plan.filters == {"asset_bucket": "Equity", "asset_class": "Equity"}
    assert plan.numeric_filters == {"holding_period_months": {"gte": 27.0, "lt": 100.0}}
    assert plan.group_by == ["holder_name"]
    assert plan.sort == {"field": "xirr_percent", "direction": "desc"}
    assert {row["scheme_name"] for row in rows} == {"One Best In Range", "Two Best In Range"}
    assert "For Holder One, One Best In Range has the highest xirr_percent at 13.8%" in answer
    assert "For Holder Two, Two Best In Range has the highest xirr_percent at 10.95%" in answer


def test_lowest_xirr_with_holding_period_range_ranks_by_holder():
    context = {
        "mutual_fund_holdings": {
            "rows": [
                {
                    "holder_name": "Holder One",
                    "scheme_name": "One Lowest In Range",
                    "asset_bucket": "Equity",
                    "asset_class": "Equity",
                    "xirr_percent": -9.55,
                    "holding_period_months": 9,
                },
                {
                    "holder_name": "Holder One",
                    "scheme_name": "One Excluded Too Short",
                    "asset_bucket": "Equity",
                    "asset_class": "Equity",
                    "xirr_percent": -20.0,
                    "holding_period_months": 7,
                },
                {
                    "holder_name": "Holder One",
                    "scheme_name": "One Higher In Range",
                    "asset_bucket": "Equity",
                    "asset_class": "Equity",
                    "xirr_percent": 4.1,
                    "holding_period_months": 22,
                },
                {
                    "holder_name": "Holder Two",
                    "scheme_name": "Two Lowest In Range",
                    "asset_bucket": "Equity",
                    "asset_class": "Equity",
                    "xirr_percent": 7.18,
                    "holding_period_months": 26,
                },
                {
                    "holder_name": "Holder Two",
                    "scheme_name": "Two Excluded Too Long",
                    "asset_bucket": "Equity",
                    "asset_class": "Equity",
                    "xirr_percent": -5.0,
                    "holding_period_months": 120,
                },
            ]
        }
    }

    question = (
        "Which fund has had the lowest XIRR return in Equity which is atleast "
        "8 months and above but less than 100 months?"
    )
    plan, result = _execute(question, context)
    answer = _structured_fallback_answer(question, result)

    rows = result["datasets"]["mutual_fund_holdings"]["rows"]
    assert plan.datasets == ["mutual_fund_holdings"]
    assert plan.filters == {"asset_bucket": "Equity", "asset_class": "Equity"}
    assert plan.numeric_filters == {"holding_period_months": {"gte": 8.0, "lt": 100.0}}
    assert plan.group_by == ["holder_name"]
    assert plan.sort == {"field": "xirr_percent", "direction": "asc"}
    assert {row["scheme_name"] for row in rows} == {"One Lowest In Range", "Two Lowest In Range"}
    assert "For Holder One, One Lowest In Range has the lowest xirr_percent at -9.55%" in answer
    assert "For Holder Two, Two Lowest In Range has the lowest xirr_percent at 7.18%" in answer


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


def test_longest_held_funds_individually_ranks_within_each_holder():
    context = {
        "mutual_fund_holdings": {
            "rows": [
                {
                    "holder_name": "Holder One",
                    "scheme_name": "One Short Fund",
                    "holding_period_months": 10,
                },
                {
                    "holder_name": "Holder One",
                    "scheme_name": "One Long Fund",
                    "holding_period_months": 100,
                },
                {
                    "holder_name": "Holder Two",
                    "scheme_name": "Two Short Fund",
                    "holding_period_months": 20,
                },
                {
                    "holder_name": "Holder Two",
                    "scheme_name": "Two Long Fund",
                    "holding_period_months": 80,
                },
            ]
        }
    }

    plan, result = _execute(
        "Which fund has been held longest for Holder One and Holder Two individually?",
        context,
    )
    answer = _structured_fallback_answer(
        "Which fund has been held longest for Holder One and Holder Two individually?",
        result,
    )

    rows = result["datasets"]["mutual_fund_holdings"]["rows"]
    assert plan.datasets == ["mutual_fund_holdings"]
    assert plan.filters == {}
    assert plan.group_by == ["holder_name"]
    assert plan.sort == {"field": "holding_period_months", "direction": "desc"}
    assert {row["scheme_name"] for row in rows} == {"One Long Fund", "Two Long Fund"}
    assert answer == (
        "For Holder One, One Long Fund has been held longest at 100 months; "
        "For Holder Two, Two Long Fund has been held longest at 80 months."
    )


def test_longest_held_assets_individually_ranks_asset_rows_within_each_holder():
    context = {
        "asset_allocations": {
            "rows": [],
            "holder_rows": [
                {
                    "holder_name": "Holder One",
                    "asset_class": "Equity",
                    "holding_period_months": 20,
                    "current_value": 100.0,
                },
                {
                    "holder_name": "Holder One",
                    "asset_class": "Debt/Fixed Income",
                    "holding_period_months": 50,
                    "current_value": 200.0,
                },
                {
                    "holder_name": "Holder Two",
                    "asset_class": "Equity",
                    "holding_period_months": 70,
                    "current_value": 300.0,
                },
                {
                    "holder_name": "Holder Two",
                    "asset_class": "Hybrid",
                    "holding_period_months": 30,
                    "current_value": 400.0,
                },
            ],
        }
    }

    plan, result = _execute(
        "Which asset has been held longest for Holder One and Holder Two individually?",
        context,
    )
    answer = _structured_fallback_answer(
        "Which asset has been held longest for Holder One and Holder Two individually?",
        result,
    )

    rows = result["datasets"]["asset_allocations"]["rows"]
    assert plan.datasets == ["asset_allocations"]
    assert plan.filters == {}
    assert plan.group_by == ["holder_name"]
    assert {row["asset_class"] for row in rows} == {"Debt/Fixed Income", "Equity"}
    assert "For Holder One, Debt/Fixed Income has been held longest at 50 months" in answer
    assert "For Holder Two, Equity has been held longest at 70 months" in answer


def test_asset_allocation_wise_family_and_individual_uses_allocation_rows():
    context = {
        "asset_allocations": {
            "rows": [
                {
                    "asset_class": "Debt/Fixed Income",
                    "asset_bucket": "Debt",
                    "current_value": 300.0,
                    "current_allocation_percent": 60.0,
                },
                {
                    "asset_class": "Equity",
                    "asset_bucket": "Equity",
                    "current_value": 100.0,
                    "current_allocation_percent": 20.0,
                },
                {
                    "asset_class": "Hybrid",
                    "asset_bucket": "Equity",
                    "current_value": 100.0,
                    "current_allocation_percent": 20.0,
                },
            ],
        },
        "holder_asset_allocations": {
            "rows": [
                {
                    "holder_name": "Holder One",
                    "asset_class": "Debt/Fixed Income",
                    "asset_bucket": "Debt",
                    "current_value": 200.0,
                    "current_allocation_percent": 80.0,
                },
                {
                    "holder_name": "Holder One",
                    "asset_class": "Hybrid",
                    "asset_bucket": "Equity",
                    "current_value": 50.0,
                    "current_allocation_percent": 20.0,
                },
                {
                    "holder_name": "Holder Two",
                    "asset_class": "Equity",
                    "asset_bucket": "Equity",
                    "current_value": 50.0,
                    "current_allocation_percent": 100.0,
                },
            ],
        },
        "mutual_fund_holdings": {
            "rows": [
                {
                    "holder_name": "Holder One",
                    "scheme_name": "Debt Fund",
                    "asset_bucket": "Debt",
                    "current_value": 200.0,
                },
            ],
        },
    }

    question = (
        "Display my Portfolio as a Family and as per individual Asset Allocation wise "
        "(Asset Allocation is a display of funds in the form of Equity, Debt, Hybrid, "
        "Balanced Advantage, International)"
    )
    plan, result = _execute(question, context)
    answer = _structured_fallback_answer(question, result)

    assert plan.datasets == ["asset_allocations", "holder_asset_allocations"]
    assert plan.filters == {}
    assert plan.group_by == []
    assert result["datasets"]["asset_allocations"]["matched_rows"] == 3
    assert result["datasets"]["holder_asset_allocations"]["matched_rows"] == 3
    assert "Family Allocation: Debt/Fixed Income: current_value 300.0, allocation 60.0%" in answer
    assert "Hybrid: current_value 100.0, allocation 20.0%" in answer
    assert "Individual Allocation: Holder One:" in answer
    assert "Holder Two: Equity: current_value 50.0, allocation 100.0%" in answer


def test_asset_class_segregation_uses_sub_asset_rows_for_family_and_individual():
    context = {
        "sub_asset_allocations": {
            "rows": [
                {
                    "asset_class": "Equity",
                    "sub_asset_class": "Large Cap",
                    "asset_bucket": "Equity",
                    "current_value": 100.0,
                    "current_allocation_percent": 10.0,
                },
                {
                    "asset_class": "Hybrid",
                    "sub_asset_class": "Dynamic Asset Allocation or Balanced Advantage",
                    "asset_bucket": "Equity",
                    "current_value": 200.0,
                    "current_allocation_percent": 20.0,
                },
                {
                    "asset_class": "Debt/Fixed Income",
                    "sub_asset_class": "Banking and PSU",
                    "asset_bucket": "Debt",
                    "current_value": 300.0,
                    "current_allocation_percent": 30.0,
                },
            ],
            "holder_rows": [
                {
                    "holder_name": "Holder One",
                    "asset_class": "Equity",
                    "sub_asset_class": "Large Cap",
                    "asset_bucket": "Equity",
                    "current_value": 40.0,
                    "current_allocation_percent": 8.0,
                },
                {
                    "holder_name": "Holder One",
                    "asset_class": "Debt/Fixed Income",
                    "sub_asset_class": "Banking and PSU",
                    "asset_bucket": "Debt",
                    "current_value": 60.0,
                    "current_allocation_percent": 12.0,
                },
                {
                    "holder_name": "Holder Two",
                    "asset_class": "Hybrid",
                    "sub_asset_class": "Multi Asset Allocation",
                    "asset_bucket": "Equity",
                    "current_value": 80.0,
                    "current_allocation_percent": 16.0,
                },
            ],
        },
        "mutual_fund_holdings": {
            "rows": [
                {
                    "holder_name": "Holder One",
                    "scheme_name": "Debt Fund",
                    "asset_bucket": "Debt",
                    "current_value": 60.0,
                }
            ]
        },
    }
    question = (
        "Display my Portfolio as a Family and as per individual in an Asset class segregation investment "
        "Asset class is Equity (Large Cap, Mid Cap, Flexi Cap, Balance Advantage, Multi Asset) "
        "Asset class is Debt (Banking and PSU, Corporate Bond Funds, Credit Risk Funds)"
    )
    plan, result = _execute(question, context)
    answer = _asset_class_segregation_answer(question, context)

    assert plan.datasets == ["sub_asset_allocations"]
    assert plan.filters == {}
    assert plan.group_by == []
    assert result["datasets"]["sub_asset_allocations"]["matched_rows"] == 3
    assert "Family Asset Class Segregation:" in answer
    assert "Equity / Large Cap: current_value 100.0, allocation 10.0%" in answer
    assert "Debt/Fixed Income / Banking and PSU: current_value 300.0, allocation 30.0%" in answer
    assert "Individual Asset Class Segregation: Holder One:" in answer
    assert "Holder Two: Hybrid / Multi Asset Allocation: current_value 80.0, allocation 16.0%" in answer


def test_sub_asset_allocation_compares_parent_and_overall_family_portfolio():
    context = {
        "asset_allocations": {
            "rows": [
                {
                    "asset_class": "Debt/Fixed Income",
                    "asset_bucket": "Debt",
                    "current_value": 396696043.0,
                },
                {
                    "asset_class": "Equity",
                    "asset_bucket": "Equity",
                    "current_value": 245974327.0,
                },
            ],
        }
    }
    plan = {
        "filters": {
            "asset_bucket": "Debt",
            "sub_asset_class": "Banking and PSU",
        }
    }
    data = {
        "datasets": {
            "sub_asset_allocations": {
                "rows": [
                    {
                        "asset_class": "Debt/Fixed Income",
                        "sub_asset_class": "Banking and PSU",
                        "asset_bucket": "Debt",
                        "current_value": 91503613.0,
                        "current_allocation_percent": 14.24,
                    }
                ]
            }
        }
    }

    answer = _sub_asset_parent_allocation_answer(
        "What is the Banking and PSU Allocation in % terms as compared to the Debt Portion and overall portfolio?",
        plan,
        data,
        context,
    )

    assert answer == (
        "Banking and PSU is 23.07% of Debt/Fixed Income "
        "and 14.24% of the overall portfolio."
    )


def test_sub_asset_allocation_compares_parent_and_overall_holder_portfolio():
    context = {
        "asset_allocations": {
            "holder_rows": [
                {
                    "holder_name": "Holder One",
                    "asset_class": "Debt/Fixed Income",
                    "asset_bucket": "Debt",
                    "current_value": 28760111.0,
                },
                {
                    "holder_name": "Holder One",
                    "asset_class": "Equity",
                    "asset_bucket": "Equity",
                    "current_value": 48555269.0,
                },
            ],
        }
    }
    plan = {
        "filters": {
            "holder_name": "Holder One",
            "asset_bucket": "Debt",
            "sub_asset_class": "Banking and PSU",
        }
    }
    data = {
        "datasets": {
            "sub_asset_allocations": {
                "rows": [
                    {
                        "holder_name": "Holder One",
                        "asset_class": "Debt/Fixed Income",
                        "sub_asset_class": "Banking and PSU",
                        "asset_bucket": "Debt",
                        "current_value": 15194973.0,
                        "current_allocation_percent": 19.65,
                    }
                ]
            }
        }
    }

    answer = _sub_asset_parent_allocation_answer(
        "What is the Banking and PSU Allocation in % terms as compared to the Debt Portion and overall portfolio for Holder One?",
        plan,
        data,
        context,
    )

    assert answer == (
        "For Holder One, Banking and PSU is 52.83% of Debt/Fixed Income "
        "and 19.65% of the overall portfolio."
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


def test_family_total_assets_does_not_filter_to_one_roowalla_holder():
    context = {
        "holder_returns": {
            "rows": [
                {
                    "holder_name": "SULEMANJI M ROOWALA",
                    "current_value": 565389857.0,
                },
                {
                    "holder_name": "Ms. ROOWALLA FATEMA SULEMANJI",
                    "current_value": 77280470.0,
                },
            ]
        }
    }

    for question in (
        "what are the total assets for the roowalla family",
        "total networth of the roowalla family",
    ):
        plan, result = _execute(question, context)

        assert plan.datasets == ["holder_returns"]
        assert plan.filters == {}
        assert plan.metrics[0] == "current_value"
        assert result["datasets"]["holder_returns"]["totals"]["current_value"] == 642670327.0


def test_debt_to_equity_ratio_uses_asset_buckets_not_conflicting_filters():
    context = {
        "asset_allocations": {
            "rows": [
                {
                    "asset_bucket": "Debt",
                    "asset_class": "Debt/Fixed Income",
                    "current_value": 300.0,
                },
                {
                    "asset_bucket": "Equity",
                    "asset_class": "Equity",
                    "current_value": 100.0,
                },
            ],
            "holder_rows": [
                {
                    "holder_name": "Holder A",
                    "asset_bucket": "Debt",
                    "asset_class": "Debt/Fixed Income",
                    "current_value": 200.0,
                },
                {
                    "holder_name": "Holder A",
                    "asset_bucket": "Equity",
                    "asset_class": "Equity",
                    "current_value": 50.0,
                },
            ],
        }
    }

    plan, result = _execute(
        "What is the debt to equity ratio in terms of percentage?",
        context,
    )
    answer = _structured_fallback_answer(
        "What is the debt to equity ratio in terms of percentage?",
        result,
    )

    assert plan.datasets == ["asset_allocations"]
    assert plan.filters == {"asset_bucket": ["Debt", "Equity"]}
    assert plan.group_by == ["asset_bucket"]
    assert result["datasets"]["asset_allocations"]["matched_rows"] == 2
    assert answer == (
        "Debt is 75.0% and Equity is 25.0% of the portfolio. "
        "The debt-to-equity ratio is 300.0%."
    )


def test_average_xirr_for_equity_funds_uses_reported_allocation_xirr():
    context = {
        "asset_allocations": {
            "rows": [
                {
                    "asset_class": "Equity",
                    "asset_bucket": "Equity",
                    "xirr_percent": 10.97,
                    "current_value": 35524938.0,
                },
                {
                    "asset_class": "Debt/Fixed Income",
                    "asset_bucket": "Debt",
                    "xirr_percent": 6.85,
                    "current_value": 367935932.0,
                },
            ]
        },
        "mutual_fund_holdings": {
            "rows": [
                {
                    "scheme_name": "Equity Fund A",
                    "asset_bucket": "Equity",
                    "asset_class": "Equity",
                    "xirr_percent": 8.0,
                },
                {
                    "scheme_name": "Equity Fund B",
                    "asset_bucket": "Equity",
                    "asset_class": "Equity",
                    "xirr_percent": 12.0,
                },
                {
                    "scheme_name": "Debt Fund",
                    "asset_bucket": "Debt",
                    "asset_class": "Debt/Fixed Income",
                    "xirr_percent": 6.0,
                },
            ]
        }
    }

    plan, result = _execute(
        "What is the average XIRR return for my equity funds?",
        context,
    )
    answer = _structured_fallback_answer(
        "What is the average XIRR return for my equity funds?",
        result,
    )

    assert plan.datasets == ["asset_allocations"]
    assert plan.filters == {"asset_bucket": "Equity", "asset_class": "Equity"}
    assert plan.metrics == ["xirr_percent"]
    assert result["datasets"]["asset_allocations"]["matched_rows"] == 1
    assert answer == "PDF-reported XIRR for Equity is 10.97%."


def test_structured_fallback_answer_uses_totals_for_multiple_rows():
    answer = _structured_fallback_answer(
        "what are the total assets for the roowalla family",
        {
            "plan": {"metrics": ["current_value"]},
            "datasets": {
                "holder_returns": {
                    "rows": [
                        {"holder_name": "A", "current_value": 100.0},
                        {"holder_name": "B", "current_value": 50.0},
                    ],
                    "totals": {"current_value": 150.0},
                }
            },
        },
    )

    assert answer == "total current_value is 150.0."
