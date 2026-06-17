from decimal import Decimal

from cortexweave_core.rag.pipelines.portfolio.mutual_fund_value_extractor import (
    extract_mutual_fund_values,
)


def test_extract_splits_fused_mf_percent_and_three_digit_purchase_price():
    values = extract_mutual_fund_values(
        "29,576.54943,88,246 89,10,823 45,22,5750 13.80 "
        "83,42,470 1.58148.369 301.280 103.0677 38,19,895 1071210544"
    )

    assert values["percent_to_mf_portfolio"] == Decimal("1.58")
    assert values["purchase_price"] == Decimal("148.369")
    assert values["nav"] == Decimal("301.280")
