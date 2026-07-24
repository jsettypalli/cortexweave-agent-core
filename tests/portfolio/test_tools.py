import asyncio
import json
from types import SimpleNamespace

import pytest

from cortexweave_core.rag.pipelines.portfolio import tools as portfolio_tools
from cortexweave_core.rag.pipelines.portfolio.tools import (
    MAX_AGENT_TABLE_PAYLOAD_BYTES,
    _compact_agent_table_result,
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


def test_agent_table_transport_is_generic_compact_and_not_duplicated():
    rows = [
        {
            "future_id": idx,
            "items": [{"name": f"Item {item_idx}", "description": "x" * 100} for item_idx in range(8)],
        }
        for idx in range(20)
    ]
    result = {
        "answer": "Future table",
        "intent": "future_intent",
        "data": {"datasets": {"future_table": {"rows": rows}}},
        "sources": [{"large": "source"}],
        "warnings": [],
        "tables": [{
            "id": "future_table",
            "title": "Future Table",
            "display": "modal",
            "columns": [
                {"key": "future_id", "label": "ID", "type": "number"},
                {"key": "items", "label": "Items", "type": "list"},
            ],
            "rows": rows,
            "total_rows": 20,
            "rows_complete": True,
            "row_key_fields": ["future_id"],
        }],
    }

    compact = _compact_agent_table_result(result)
    table = compact["tables"][0]

    assert compact["data"] == {}
    assert compact["sources"] == []
    assert len(table["rows"]) == 10
    assert len(table["rows"][0]["items"]) == 3
    assert table["rows"][0]["_nested_counts"]["items"] == 8
    assert table["rows_complete"] is False
    assert table["nested_fields"]["items"]["complete"] is False
    assert len(json.dumps(compact).encode("utf-8")) <= MAX_AGENT_TABLE_PAYLOAD_BYTES


def test_agent_fund_overlap_warning_describes_lazy_loaded_stocks():
    result = {
        "answer": "Fund overlap",
        "intent": "fund_overlap_matrix",
        "data": {},
        "sources": [],
        "warnings": [],
        "tables": [{
            "id": "fund_overlap_matrix",
            "title": "Fund Overlap Matrix",
            "display": "matrix",
            "columns": [
                {"key": "fund_a", "label": "Fund A", "type": "text"},
                {"key": "fund_b", "label": "Fund B", "type": "text"},
                {"key": "stocks", "label": "Stocks", "type": "list"},
            ],
            "rows": [{
                "fund_a": "Fund A",
                "fund_b": "Fund B",
                "shared_stocks": 44,
                "stocks": [{"name": "Stock 1"}],
            }],
            "total_rows": 1,
            "rows_complete": True,
            "nested_fields": {
                "stocks": {"initial_limit": 0, "total_key": "shared_stocks", "complete": False},
            },
            "warnings": [
                "Shared stock lists are capped to top 10 by portfolio exposure in this view.",
            ],
        }],
    }

    table = _compact_agent_table_result(result)["tables"][0]

    assert table["warnings"] == [
        "Additional table details load when the table or row is opened.",
    ]


@pytest.mark.parametrize(
    "table_id",
    ["fund_resolution_status", "sector_overlap", "fund_overlap_matrix", "stock_overlap", "future_table"],
)
def test_structured_table_tool_response_skips_model_summarization(monkeypatch, table_id):
    async def fake_query(_question):
        return {
            "answer": "Structured result",
            "intent": table_id,
            "data": {},
            "sources": [],
            "warnings": [],
            "tables": [{
                "id": table_id,
                "title": table_id,
                "display": "modal",
                "columns": [],
                "rows": [],
            }],
        }

    monkeypatch.setattr(portfolio_tools, "_answer_portfolio_query", fake_query)
    context = SimpleNamespace(actions=SimpleNamespace(skip_summarization=False))

    result = asyncio.run(portfolio_tools.answer_portfolio_query("question", context))

    assert context.actions.skip_summarization is True
    assert result["tables"][0]["id"] == table_id


def test_non_table_tool_response_keeps_normal_model_path(monkeypatch):
    expected = {"answer": "Plain answer", "intent": "summary", "tables": []}

    async def fake_query(_question):
        return expected

    monkeypatch.setattr(portfolio_tools, "_answer_portfolio_query", fake_query)
    context = SimpleNamespace(actions=SimpleNamespace(skip_summarization=False))

    result = asyncio.run(portfolio_tools.answer_portfolio_query("question", context))

    assert context.actions.skip_summarization is False
    assert result is expected
