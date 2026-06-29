from dataclasses import dataclass, field
import json
import re
from difflib import SequenceMatcher
from typing import Any, Protocol


class GenerationModel(Protocol):
    def generate_response(self, prompt: str, context: list[Any]) -> Any: ...


@dataclass
class PortfolioQueryPlan:
    intent: str
    datasets: list[str]
    filters: dict[str, str | list[str]] = field(default_factory=dict)
    metrics: list[str] = field(default_factory=list)
    group_by: list[str] = field(default_factory=list)
    sort: dict[str, str] | None = None
    limit: int | None = None
    requires_external_metadata: bool = False
    external_metadata_reason: str | None = None
    planner_source: str = "heuristic"

    def to_dict(self) -> dict[str, Any]:
        return {
            "intent": self.intent,
            "datasets": self.datasets,
            "filters": self.filters,
            "metrics": self.metrics,
            "group_by": self.group_by,
            "sort": self.sort,
            "limit": self.limit,
            "requires_external_metadata": self.requires_external_metadata,
            "external_metadata_reason": self.external_metadata_reason,
            "planner_source": self.planner_source,
        }


DATASET_ALIASES = {
    "asset_allocations": ("asset", "allocation", "allocations", "exposure", "portfolio", "mix"),
    "sub_asset_allocations": ("sub asset", "category", "categories", "cap"),
    "mutual_fund_holdings": ("mutual fund", "mutual funds", "fund", "funds", "fund name", "fund names", "scheme", "schemes"),
    "holder_returns": ("return", "returns", "xirr", "gain", "gains", "profit", "performance", "invested"),
    "pms_holdings": ("pms", "portfolio management"),
    "bond_holdings": ("bond", "bonds", "fixed deposit", "fixed deposits", "fd", "fds"),
}

METRIC_ALIASES = {
    "current_value": ("current value", "market value", "value", "worth", "networth", "exposure"),
    "invested_value": ("invested", "investment", "cost"),
    "xirr_percent": ("xirr", "return", "returns", "performance"),
    "current_allocation_percent": ("allocation", "allocation percent", "percentage", "%"),
    "total_gain": ("total gain", "gain", "gains", "profit"),
    "total_returns": ("total returns",),
    "gain_loss_percent": ("gain percent", "gain percentage", "gain/loss percent", "gain loss percent"),
    "realized_gain_loss": ("realized", "realised"),
    "unrealized_gain_loss": ("unrealized", "unrealised"),
    "dividend_interest_paid": ("dividend", "interest paid", "interest"),
    "holding_period_months": ("holding period", "held", "longest", "months"),
    "nav": ("nav", "net asset value"),
    "purchase_price": ("purchase price", "purchase nav"),
}

EXTERNAL_METADATA_TERMS = (
    "company",
    "companies",
    "stock",
    "stocks",
    "underlying",
    "constituents",
    "holdings inside",
    "fund holdings",
    "riskometer",
    "risk-o-meter",
    "rating",
    "ratings",
    "peer",
    "peers",
)

ALLOWED_DATASETS = {
    "asset_allocations",
    "holder_asset_allocations",
    "sub_asset_allocations",
    "mutual_fund_holdings",
    "holder_returns",
    "pms_holdings",
    "bond_holdings",
}

ALLOWED_METRICS = set(METRIC_ALIASES)
ALLOWED_GROUP_BY = {"holder_name", "asset_bucket", "asset_class", "sub_asset_class"}
ALLOWED_FILTERS = {"holder_name", "asset_bucket", "asset_class", "sub_asset_class", "scheme_name", "security_name"}

AMBIGUOUS_DIMENSION_TERMS = {
    "value",
}

LOW_SIGNAL_DIMENSION_TOKENS = {
    "asset",
    "assets",
    "allocation",
    "allocations",
    "class",
    "fund",
    "funds",
    "cap",
    "category",
    "holder",
    "holders",
}


def build_portfolio_query_plan(question: str, schema: dict[str, Any]) -> PortfolioQueryPlan:
    normalized = _normalize(question)
    filters = _infer_filters(normalized, schema)
    datasets = _infer_datasets(normalized, filters)
    metrics = _infer_metrics(normalized)
    group_by = _infer_group_by(normalized)
    sort = _infer_sort(normalized)
    limit = _infer_limit(normalized)
    requires_external_metadata, reason = _external_metadata_requirement(normalized)
    if _asks_for_broad_asset_bucket_distribution(normalized):
        filters.pop("asset_class", None)
        filters["asset_bucket"] = _requested_asset_buckets(normalized)
        group_by = [field for field in group_by if field != "asset_class"]
        group_by = _dedupe([*group_by, "asset_bucket"])
    if _asks_for_holder_ranking(normalized):
        filters.pop("holder_name", None)
        datasets = ["holder_returns"]
        group_by = []
    if _asks_for_family_total_assets(normalized):
        filters.pop("holder_name", None)
        datasets = ["holder_returns"]
        group_by = []
        metrics = _dedupe(["current_value", *metrics])

    if _asks_for_reported_average_xirr(normalized) and (
        filters.get("asset_bucket") or filters.get("asset_class")
    ):
        datasets = ["asset_allocations"]
        metrics = ["xirr_percent"]

    has_fund_holdings = bool(schema.get("mutual_fund_schemes"))
    asks_for_fund_holdings = any(term in normalized for term in ("fund name", "fund names", "scheme", "schemes", "mutual fund", "mutual funds"))
    asks_for_funds = asks_for_fund_holdings or "fund" in _tokens(normalized) or "funds" in _tokens(normalized)
    asks_for_funds = asks_for_funds or bool(filters.get("scheme_name"))
    if not (has_fund_holdings and asks_for_funds):
        datasets = [dataset for dataset in datasets if dataset != "mutual_fund_holdings"]
    if (
        has_fund_holdings
        and asks_for_funds
        and not _asks_for_reported_average_xirr(normalized)
    ):
        datasets = [dataset for dataset in datasets if dataset not in {"holder_returns", "asset_allocations", "sub_asset_allocations"}]
        if "mutual_fund_holdings" not in datasets:
            datasets.insert(0, "mutual_fund_holdings")
    elif filters.get("sub_asset_class") and "sub_asset_allocations" not in datasets:
        datasets.insert(0, "sub_asset_allocations")
    if filters.get("sub_asset_class") and "mutual_fund_holdings" not in datasets:
        datasets = [dataset for dataset in datasets if dataset != "asset_allocations"]
    if filters.get("scheme_name") and "mutual_fund_holdings" not in datasets:
        datasets.insert(0, "mutual_fund_holdings")
    if filters.get("asset_bucket") and not datasets:
        datasets.append("asset_allocations")
    if filters.get("asset_class") and not datasets:
        datasets.append("asset_allocations")
    sort = _question_sort_override(normalized) or sort

    if not datasets:
        datasets = ["asset_allocations", "sub_asset_allocations", "mutual_fund_holdings", "holder_returns", "pms_holdings", "bond_holdings"]

    if not metrics:
        metrics = ["current_value", "invested_value", "current_allocation_percent", "xirr_percent", "total_gain"]

    return PortfolioQueryPlan(
        intent="planned_portfolio_query",
        datasets=_dedupe(datasets),
        filters=filters,
        metrics=metrics,
        group_by=group_by,
        sort=sort,
        limit=limit,
        requires_external_metadata=requires_external_metadata,
        external_metadata_reason=reason,
    )


def build_model_assisted_portfolio_query_plan(
    generation_model: GenerationModel | None,
    question: str,
    schema: dict[str, Any],
) -> tuple[PortfolioQueryPlan, list[str]]:
    fallback = build_portfolio_query_plan(question, schema)
    if generation_model is None:
        return fallback, ["Model-assisted portfolio planning skipped because no generation model is configured."]

    prompt = _planner_prompt(question, schema)
    try:
        raw_response = "".join(generation_model.generate_response(prompt, []))
        payload = _extract_json_object(raw_response)
        plan = _validated_plan_from_payload(payload, schema, question)
    except Exception as exc:
        return fallback, [f"Model-assisted portfolio planning failed; heuristic plan used: {exc}"]

    if not plan.datasets:
        return fallback, ["Model-assisted portfolio planning returned no valid datasets; heuristic plan used."]
    return plan, []


def portfolio_schema_from_context(context: dict[str, Any]) -> dict[str, Any]:
    rows = []
    mutual_fund_rows = []
    for dataset_name, dataset in context.items():
        if isinstance(dataset, dict):
            rows.extend(dataset.get("rows", []))
            rows.extend(dataset.get("holder_rows", []))
            if dataset_name == "mutual_fund_holdings":
                mutual_fund_rows.extend(dataset.get("rows", []))

    return {
        "holders": sorted({row["holder_name"] for row in rows if row.get("holder_name")}),
        "asset_buckets": sorted({row["asset_bucket"] for row in rows if row.get("asset_bucket")}),
        "asset_classes": sorted({row["asset_class"] for row in rows if row.get("asset_class")}),
        "sub_asset_classes": sorted({row["sub_asset_class"] for row in rows if row.get("sub_asset_class")}),
        "schemes": sorted({row["scheme_name"] for row in rows if row.get("scheme_name")}),
        "mutual_fund_schemes": sorted({row["scheme_name"] for row in mutual_fund_rows if row.get("scheme_name")}),
        "securities": sorted({row["security_name"] for row in rows if row.get("security_name")}),
    }


def _planner_prompt(question: str, schema: dict[str, Any]) -> str:
    schema_json = json.dumps(schema, ensure_ascii=False)
    allowed = {
        "datasets": sorted(ALLOWED_DATASETS),
        "filters": sorted(ALLOWED_FILTERS),
        "metrics": sorted(ALLOWED_METRICS),
        "group_by": sorted(ALLOWED_GROUP_BY),
    }
    allowed_json = json.dumps(allowed, ensure_ascii=False)
    return f"""
You are a portfolio query planner. Convert the user's portfolio question into one JSON object.

User question:
{question}

Available extracted portfolio schema:
{schema_json}

Allowed plan values:
{allowed_json}

Rules:
- Return JSON only. No markdown.
- Use only dataset/filter/group_by/metric names listed in allowed plan values.
- Use filter values exactly as they appear in the available schema.
- Filters may be a string or a list of strings.
- If the question asks for family and individual views, use sub_asset_allocations or asset_allocations with group_by ["holder_name"] when the requested rows support holder_name.
- If the question asks for distribution between broad categories like debt and equity, use group_by ["asset_bucket"] and filter asset_bucket, not asset_class.
- If the question asks about funds/schemes/fund names or "equity funds", use mutual_fund_holdings.
- If the question asks for best/highest returns, sort by xirr_percent descending.
- If the question asks for highest/lowest gain or profit, sort by total_gain descending/ascending.
- If the question asks which holding has been held longest, sort by holding_period_months descending.
- If the question asks which holding has been held shortest, sort by holding_period_months ascending.
- If the user lists multiple asset or sub-asset classes, return all matching classes as a list filter.
- Do not match generic words like "portfolio" to scheme_name unless the user asks for PMS/scheme/holding/fund details.
- Do not match metric words like "value" from "market value" to the "Value" sub_asset_class unless the user clearly asks for Value funds/category.
- If the question asks about underlying companies, stocks, ratings, riskometer, peers, or fund constituents, set requires_external_metadata true.

Required JSON shape:
{{
  "datasets": ["sub_asset_allocations"],
  "filters": {{"asset_class": "Equity"}},
  "metrics": ["current_value", "invested_value"],
  "group_by": ["holder_name"],
  "sort": null,
  "limit": null,
  "requires_external_metadata": false,
  "external_metadata_reason": null
}}
""".strip()


def _extract_json_object(raw_response: str) -> dict[str, Any]:
    text = raw_response.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("planner response did not contain a JSON object")
    return json.loads(text[start:end + 1])


def _validated_plan_from_payload(payload: dict[str, Any], schema: dict[str, Any], question: str = "") -> PortfolioQueryPlan:
    datasets = [dataset for dataset in _as_list(payload.get("datasets")) if dataset in ALLOWED_DATASETS]
    filters = _validated_filters(payload.get("filters") or {}, schema)
    metrics = [metric for metric in _as_list(payload.get("metrics")) if metric in ALLOWED_METRICS]
    group_by = [field for field in _as_list(payload.get("group_by")) if field in ALLOWED_GROUP_BY]
    sort = payload.get("sort") if isinstance(payload.get("sort"), dict) else None
    if sort:
        sort = {
            "field": sort.get("field") if sort.get("field") in ALLOWED_METRICS else "current_value",
            "direction": "asc" if sort.get("direction") == "asc" else "desc",
        }
    limit = payload.get("limit") if isinstance(payload.get("limit"), int) and payload.get("limit") > 0 else None

    datasets, filters, group_by, sort = _repair_model_plan_for_question(question, schema, datasets, filters, group_by, sort)

    if filters.get("sub_asset_class") and "mutual_fund_holdings" in datasets:
        datasets = [dataset for dataset in datasets if dataset != "sub_asset_allocations"]
    elif filters.get("sub_asset_class") and "sub_asset_allocations" not in datasets:
        datasets.insert(0, "sub_asset_allocations")
    if filters.get("sub_asset_class") and "mutual_fund_holdings" not in datasets:
        datasets = [dataset for dataset in datasets if dataset != "asset_allocations"]
    if filters.get("asset_bucket") and not datasets:
        datasets.append("asset_allocations")
    if filters.get("asset_class") and not datasets:
        datasets.append("asset_allocations")
    if not metrics:
        metrics = ["current_value", "invested_value", "current_allocation_percent", "xirr_percent", "total_gain"]

    requires_external_metadata = bool(payload.get("requires_external_metadata"))
    external_metadata_reason = payload.get("external_metadata_reason")
    if requires_external_metadata and not external_metadata_reason:
        external_metadata_reason = "The question needs external fund metadata or underlying security holdings that are not present in Scripbox portfolio PDFs."

    return PortfolioQueryPlan(
        intent="planned_portfolio_query",
        datasets=_dedupe(datasets),
        filters=filters,
        metrics=metrics,
        group_by=group_by,
        sort=sort,
        limit=limit,
        requires_external_metadata=requires_external_metadata,
        external_metadata_reason=external_metadata_reason,
        planner_source="model",
    )


def _repair_model_plan_for_question(
    question: str,
    schema: dict[str, Any],
    datasets: list[str],
    filters: dict[str, str | list[str]],
    group_by: list[str],
    sort: dict[str, str] | None,
) -> tuple[list[str], dict[str, str | list[str]], list[str], dict[str, str] | None]:
    normalized = _normalize(question)
    has_fund_holdings = bool(schema.get("mutual_fund_schemes"))
    asks_for_funds = any(term in normalized for term in ("fund", "funds", "scheme", "schemes", "mutual fund", "mutual funds"))
    if _asks_for_reported_average_xirr(normalized) and (filters.get("asset_bucket") or filters.get("asset_class")):
        datasets = ["asset_allocations"]
    elif has_fund_holdings and asks_for_funds:
        datasets = [dataset for dataset in datasets if dataset not in {"holder_returns", "asset_allocations", "sub_asset_allocations"}]
        if "mutual_fund_holdings" not in datasets:
            datasets.insert(0, "mutual_fund_holdings")
    if _asks_for_broad_asset_bucket_distribution(normalized):
        filters.pop("asset_class", None)
        filters["asset_bucket"] = _requested_asset_buckets(normalized)
        group_by = [field for field in group_by if field != "asset_class"]
        group_by = _dedupe([*group_by, "asset_bucket"])
        if not datasets:
            datasets.append("asset_allocations")

    sort = _question_sort_override(normalized) or sort

    return datasets, filters, group_by, sort


def _validated_filters(raw_filters: dict[str, Any], schema: dict[str, Any]) -> dict[str, str | list[str]]:
    filters = {}
    schema_keys = {
        "holder_name": "holders",
        "asset_bucket": "asset_buckets",
        "asset_class": "asset_classes",
        "sub_asset_class": "sub_asset_classes",
        "scheme_name": "schemes",
        "security_name": "securities",
    }
    for key, raw_value in raw_filters.items():
        if key not in ALLOWED_FILTERS:
            continue
        allowed_values = set(schema.get(schema_keys[key], []))
        values = [value for value in _as_list(raw_value) if value in allowed_values]
        if values:
            filters[key] = values if len(values) > 1 else values[0]
    return filters


def _as_list(value: Any) -> list:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def _infer_datasets(normalized: str, filters: dict[str, str]) -> list[str]:
    datasets = []
    for dataset, aliases in DATASET_ALIASES.items():
        if any(alias in normalized for alias in aliases):
            datasets.append(dataset)

    if filters.get("sub_asset_class"):
        datasets.append("sub_asset_allocations")
    if filters.get("asset_bucket") and "sub_asset_allocations" not in datasets:
        datasets.append("asset_allocations")
    if filters.get("asset_class") and "sub_asset_allocations" not in datasets:
        datasets.append("asset_allocations")
    if "holder_name" in filters and any(term in normalized for term in ("return", "xirr", "gain", "performance")):
        datasets.append("holder_returns")
    if filters.get("scheme_name"):
        datasets.append("mutual_fund_holdings")
    return _dedupe(datasets)


def _asks_for_reported_average_xirr(normalized: str) -> bool:
    return (
        any(term in normalized for term in ("average", "avg", "mean"))
        and any(term in normalized for term in ("xirr", "return", "returns", "performance"))
    )


def _infer_filters(normalized: str, schema: dict[str, Any]) -> dict[str, str | list[str]]:
    filters = {}
    for key, schema_key in (
        ("holder_name", "holders"),
        ("asset_bucket", "asset_buckets"),
        ("asset_class", "asset_classes"),
        ("sub_asset_class", "sub_asset_classes"),
        ("scheme_name", "schemes"),
        ("security_name", "securities"),
    ):
        if key in {"scheme_name", "security_name"} and not _should_match_holding_dimension(normalized, key):
            continue
        matches = (
            _exact_dimension_matches(normalized, schema.get(schema_key, []), key)
            if key in {"scheme_name", "security_name"}
            else _dimension_matches(normalized, schema.get(schema_key, []), key)
        )
        if key == "sub_asset_class":
            asset_class_values = set(schema.get("asset_classes", []))
            matches = [match for match in matches if match not in asset_class_values]
        if not matches:
            continue
        filters[key] = matches if key in {"sub_asset_class", "asset_class"} and len(matches) > 1 else matches[0]
    return filters


def _infer_metrics(normalized: str) -> list[str]:
    metrics = []
    for metric, aliases in METRIC_ALIASES.items():
        if any(alias in normalized for alias in aliases):
            metrics.append(metric)
    return _dedupe(metrics)


def _infer_group_by(normalized: str) -> list[str]:
    group_by = []
    asks_for_holder_view = any(term in normalized for term in ("by holder", "holder wise", "holder-wise", "each holder", "individual"))
    if asks_for_holder_view:
        group_by.append("holder_name")
    if _asks_for_broad_asset_bucket_distribution(normalized):
        group_by.append("asset_bucket")
    if any(term in normalized for term in ("by asset", "asset wise", "asset-wise", "per asset", "as per asset")):
        group_by.append("asset_class")
    if any(term in normalized for term in ("by category", "by sub asset", "category wise", "sub asset wise")):
        group_by.append("sub_asset_class")
    if (
        not asks_for_holder_view
        and any(term in normalized for term in ("distribution", "segregation", "breakdown", "split"))
        and any(term in normalized for term in ("large cap", "mid cap", "small cap", "flexi cap", "multi cap", "sub asset"))
    ):
        group_by.append("sub_asset_class")
    return group_by


def _infer_sort(normalized: str) -> dict[str, str] | None:
    contextual_sort = _question_sort_override(normalized)
    if contextual_sort:
        return contextual_sort
    if any(term in normalized for term in ("highest", "largest", "top", "maximum", "biggest")):
        return {"field": "current_value", "direction": "desc"}
    if any(term in normalized for term in ("lowest", "smallest", "minimum")):
        return {"field": "current_value", "direction": "asc"}
    return None


def _question_sort_override(normalized: str) -> dict[str, str] | None:
    high_terms = ("best", "highest", "largest", "top", "maximum", "biggest")
    low_terms = ("lowest", "smallest", "minimum", "least", "worst", "shortest")
    has_high = any(term in normalized for term in high_terms)
    has_low = any(term in normalized for term in low_terms)
    if any(term in normalized for term in ("held shortest", "shortest held", "held the shortest", "held least", "least held")):
        return {"field": "holding_period_months", "direction": "asc"}
    if any(term in normalized for term in ("held longest", "longest held", "holding period", "held the longest")):
        return {"field": "holding_period_months", "direction": "desc"}
    if any(term in normalized for term in ("gain percent", "gain percentage", "gain/loss percent", "gain loss percent")):
        if has_low:
            return {"field": "gain_loss_percent", "direction": "asc"}
        if has_high:
            return {"field": "gain_loss_percent", "direction": "desc"}
    if any(term in normalized for term in ("gain", "gains", "profit", "profits")):
        if has_low:
            return {"field": "total_gain", "direction": "asc"}
        if has_high:
            return {"field": "total_gain", "direction": "desc"}
    if any(term in normalized for term in ("return", "returns", "xirr", "performance")):
        if has_low:
            return {"field": "xirr_percent", "direction": "asc"}
        if has_high:
            return {"field": "xirr_percent", "direction": "desc"}
    return None


def _infer_limit(normalized: str) -> int | None:
    match = re.search(r"\btop\s+(\d+)\b", normalized)
    return int(match.group(1)) if match else None


def _asks_for_broad_asset_bucket_distribution(normalized: str) -> bool:
    return (
        "debt" in normalized
        and "equity" in normalized
        and any(term in normalized for term in ("distribution", "between", "split", "allocation", "mix", "breakdown", "ratio"))
    )


def _asks_for_holder_ranking(normalized: str) -> bool:
    holder_terms = ("which holder", "what holder", "holder has", "holder with")
    ranking_terms = (
        "highest",
        "largest",
        "top",
        "maximum",
        "biggest",
        "lowest",
        "smallest",
        "minimum",
        "least",
        "worst",
    )
    metric_terms = ("gain", "gains", "return", "returns", "xirr", "performance")
    return (
        any(term in normalized for term in holder_terms)
        and any(term in normalized for term in ranking_terms)
        and any(term in normalized for term in metric_terms)
    )


def _asks_for_family_total_assets(normalized: str) -> bool:
    family_terms = ("family", "household", "overall", "total assets", "net worth", "networth")
    total_terms = ("total", "overall", "aggregate", "combined", "net worth", "networth")
    asset_terms = ("asset", "assets", "portfolio value", "net worth", "networth", "current value", "worth")
    return (
        any(term in normalized for term in family_terms)
        and any(term in normalized for term in total_terms)
        and any(term in normalized for term in asset_terms)
    )


def _requested_asset_buckets(normalized: str) -> list[str]:
    buckets = []
    if "debt" in normalized:
        buckets.append("Debt")
    if "equity" in normalized:
        buckets.append("Equity")
    return buckets


def _external_metadata_requirement(normalized: str) -> tuple[bool, str | None]:
    if not any(term in normalized for term in EXTERNAL_METADATA_TERMS):
        return False, None
    return (
        True,
        "The question needs external fund metadata or underlying security holdings that are not present in Scripbox portfolio PDFs.",
    )


def _dimension_matches(normalized: str, values: list[str], dimension_key: str) -> list[str]:
    normalized_values = [(value, _normalize(value)) for value in values]
    matches = []
    for value, normalized_value in normalized_values:
        if len(normalized_value) < 3:
            continue
        if _is_ambiguous_dimension_match(normalized, normalized_value, dimension_key):
            continue
        if normalized_value and normalized_value in normalized:
            matches.append(value)

    question_tokens = _meaningful_dimension_tokens(normalized)
    best_value = None
    best_score = 0.0
    for value, normalized_value in normalized_values:
        if value in matches:
            continue
        value_tokens = _meaningful_dimension_tokens(normalized_value)
        if not value_tokens:
            continue
        if _is_ambiguous_dimension_match(normalized, normalized_value, dimension_key):
            continue
        matched_tokens = [
            value_token for value_token in value_tokens
            if any(_tokens_match(question_token, value_token) for question_token in question_tokens)
        ]
        score = len(matched_tokens) / len(value_tokens)
        if dimension_key == "holder_name":
            score += _ordered_match_bonus(_meaningful_dimension_tokens(normalized), value_tokens)
        threshold = 0.25 if dimension_key == "holder_name" else 0.66
        if score >= threshold and dimension_key in {"asset_class", "sub_asset_class"}:
            matches.append(value)
        elif score > best_score and score >= threshold:
            best_value = value
            best_score = score
    if best_value:
        matches.append(best_value)
    return _dedupe(matches)


def _exact_dimension_matches(normalized: str, values: list[str], dimension_key: str) -> list[str]:
    matches = []
    for value in values:
        normalized_value = _normalize(value)
        if len(normalized_value) < 3:
            continue
        if _is_ambiguous_dimension_match(normalized, normalized_value, dimension_key):
            continue
        if normalized_value in normalized:
            matches.append(value)
    return _dedupe(matches)


def _should_match_holding_dimension(normalized: str, dimension_key: str) -> bool:
    if dimension_key == "scheme_name":
        return any(term in normalized for term in ("scheme", "fund", "pms", "holding", "nav", "purchase price", "purchase nav", "folio", "units"))
    if dimension_key == "security_name":
        return any(term in normalized for term in ("security", "bond", "fd", "fixed deposit", "holding"))
    return True


def _tokens_match(left: str, right: str) -> bool:
    return left == right or SequenceMatcher(None, left, right).ratio() >= 0.86


def _ordered_match_bonus(question_tokens: list[str], value_tokens: list[str]) -> float:
    positions = []
    for value_token in value_tokens:
        for index, question_token in enumerate(question_tokens):
            if _tokens_match(question_token, value_token):
                positions.append(index)
                break
    if len(positions) < 2:
        return 0.0
    return 0.15 if positions == sorted(positions) else 0.0


def _is_ambiguous_dimension_match(normalized: str, normalized_value: str, dimension_key: str) -> bool:
    if dimension_key not in {"asset_class", "sub_asset_class"}:
        return False
    if normalized_value not in AMBIGUOUS_DIMENSION_TERMS:
        return False
    contextual_phrases = (
        f"{normalized_value} fund",
        f"{normalized_value} funds",
        f"{normalized_value} category",
        f"{normalized_value} sub asset",
        f"{normalized_value} asset",
        f"{normalized_value} allocation",
    )
    return not any(phrase in normalized for phrase in contextual_phrases)


def _normalize(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9%&/ -]", " ", value.lower())).strip()


def _tokens(value: str) -> list[str]:
    return [_canonical_token(token) for token in re.split(r"[^a-z0-9]+", value) if token]


def _meaningful_dimension_tokens(value: str) -> list[str]:
    return [token for token in _tokens(value) if token not in LOW_SIGNAL_DIMENSION_TOKENS]


def _canonical_token(token: str) -> str:
    if token.endswith("ed") and len(token) > 4:
        return token[:-2]
    return token


def _dedupe(values: list[Any]) -> list[Any]:
    output = []
    for value in values:
        if value not in output:
            output.append(value)
    return output
