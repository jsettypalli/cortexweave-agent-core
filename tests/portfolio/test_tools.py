from cortexweave_core.rag.pipelines.portfolio.tools import (
    _compact_fund_overlap_result,
    _compact_stock_overlap_result,
)


def test_direct_fund_overlap_tool_payload_is_compacted():
    result = {
        "rows": [
            {
                "fund_a": f"Fund {idx}A",
                "fund_b": f"Fund {idx}B",
                "stocks": [
                    {"name": f"Stock {stock_idx}", "combined_portfolio_exposure_percent": 10 - stock_idx}
                    for stock_idx in range(12)
                ],
            }
            for idx in range(15)
        ],
        "totals": {"pairs_with_overlap": 15},
    }

    compact = _compact_fund_overlap_result(result, row_limit=10, stock_limit=10)

    assert len(compact["rows"]) == 10
    assert len(compact["rows"][0]["stocks"]) == 10
    assert compact["totals"]["total_rows"] == 15
    assert compact["totals"]["returned_rows"] == 10
    assert compact["totals"]["stocks_returned_per_pair"] == 10
    assert compact["warnings"] == [
        "Direct tool output is capped to 10 rows out of 15 total matches.",
        "Shared stock lists are capped to top 10 by portfolio exposure per fund pair.",
    ]


def test_direct_stock_overlap_tool_payload_is_compacted():
    result = {
        "rows": [
            {
                "stock_name": f"Stock {idx}",
                "funds": [f"Fund {fund_idx}" for fund_idx in range(12)],
            }
            for idx in range(15)
        ],
        "totals": {"overlap_count": 15},
    }

    compact = _compact_stock_overlap_result(result, row_limit=10, fund_limit=10)

    assert len(compact["rows"]) == 10
    assert len(compact["rows"][0]["funds"]) == 10
    assert compact["totals"]["total_rows"] == 15
    assert compact["totals"]["returned_rows"] == 10
    assert compact["totals"]["funds_returned_per_stock"] == 10
    assert compact["warnings"] == [
        "Direct tool output is capped to 10 rows out of 15 total matches.",
        "Fund lists are capped to top 10 entries per stock in direct tool output.",
    ]
