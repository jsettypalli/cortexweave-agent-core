import re
from decimal import Decimal, InvalidOperation

_NUMBER_PATTERN = re.compile(r"\(?-?\d[\d,]*(?:\.\d+)?\)?")


def extract_mutual_fund_values(text: str) -> dict:
    text = text.strip()
    units_cost = re.match(r"^(?P<units>\d[\d,]*\.\d{3})\s*(?P<cost>\(?\d{1,3}(?:,\d{2})+,\d{3}\)?|0)", text)
    if not units_cost:
        return {}
    values = {"units": _parse_decimal(units_cost.group("units")), "cost": _parse_decimal(units_cost.group("cost"))}
    remainder = text[units_cost.end():].strip()
    number_tokens = [m.group(0) for m in _NUMBER_PATTERN.finditer(remainder)]
    numbers = _normalize_mutual_fund_number_tokens(number_tokens)
    parsed = [_parse_decimal(n) for n in numbers]
    parsed = [n for n in parsed if n is not None]

    if parsed:
        values["market_value"] = parsed[0]
    if len(parsed) > 1:
        values["unrealized_gain_loss"] = parsed[1]
    if len(parsed) > 2:
        values["dividend_paid"] = Decimal("0") if abs(parsed[2]) > Decimal("100") else parsed[2]
    if len(parsed) > 3:
        values["xirr_percent"] = parsed[3] if abs(parsed[3]) <= Decimal("100") else None
    if len(parsed) > 4:
        values["total_returns"] = parsed[4]
    if len(parsed) > 5:
        values["percent_to_mf_portfolio"] = parsed[5] if abs(parsed[5]) <= Decimal("100") else None
    if len(parsed) > 6:
        values["purchase_price"] = parsed[6]
    if len(parsed) > 7:
        values["nav"] = parsed[7]

    if "/" in text and len(numbers) >= 4:
        folio_no = f"{numbers[-2]}/{numbers[-1]}"
        gain_loss_percent, months = _split_percent_month_token(numbers[-3])
        if gain_loss_percent is not None and months is not None:
            realized_gain_loss = parsed[-2] if len(parsed) >= 2 else Decimal("0")
        else:
            gain_loss_percent, months = _split_percent_month_token(numbers[-4])
            if gain_loss_percent is not None and months is not None:
                realized_gain_loss = parsed[-3] if len(parsed) >= 3 else Decimal("0")
            elif len(numbers) >= 5:
                gain_loss_percent = _parse_decimal(numbers[-5])
                months = _to_int(_parse_decimal(numbers[-4]))
                realized_gain_loss = parsed[-3] if len(parsed) >= 3 else Decimal("0")
            else:
                realized_gain_loss = parsed[-3] if len(parsed) >= 3 else Decimal("0")
    else:
        folio_no = numbers[-1] if numbers and re.search(r"\d{5,}", numbers[-1]) else None
        if folio_no and len(numbers) >= 2:
            gain_loss_percent, months = _split_percent_month_token(numbers[-2])
            if gain_loss_percent is not None and months is not None:
                realized_gain_loss = Decimal("0")
            else:
                gain_loss_percent, months = _split_percent_month_token(numbers[-3]) if len(numbers) >= 3 else (None, None)
                realized_gain_loss = parsed[-2] if len(parsed) >= 2 else None
        else:
            gain_loss_token = numbers[-3] if len(numbers) >= 3 else None
            gain_loss_percent, months = _split_percent_month_token(gain_loss_token) if gain_loss_token else (None, None)
            realized_gain_loss = parsed[-1] if parsed else None
    values["gain_loss_percent"] = gain_loss_percent
    values["holding_period_months"] = months
    values["realized_gain_loss"] = realized_gain_loss
    values["folio_no"] = folio_no
    return values


def _normalize_mutual_fund_number_tokens(tokens):
    if len(tokens) < 2:
        return tokens
    normalized = [tokens[0]]
    second = tokens[1]
    if re.match(r"^-?\d{1,3}(?:,\d{2})+,\d{4}$", second):
        normalized.append(second[:-1])
        normalized.append(second[-1])
        normalized.extend(tokens[2:])
        return _split_fused_percent_and_price(normalized)
    if re.match(r"^\(\d{1,3}(?:,\d{2})+,\d{3}\)0$", second):
        normalized.append(second[:-1])
        normalized.append("0")
        normalized.extend(tokens[2:])
        return _split_fused_percent_and_price(normalized)
    normalized.extend(tokens[1:])
    return _split_fused_percent_and_price(normalized)


def _split_fused_percent_and_price(tokens):
    if len(tokens) <= 6:
        return tokens
    fused_percent = tokens[5]
    next_token = tokens[6]
    match = re.fullmatch(r"(-?\d+\.\d{2})(\d{2})", fused_percent)
    if not match or not re.fullmatch(r"\d{3}", next_token):
        return tokens
    return [*tokens[:5], match.group(1), f"{match.group(2)}.{next_token}", *tokens[7:]]


def _split_percent_month_token(value):
    match = re.fullmatch(r"(-?\d+\.\d{2})(\d{1,3})", value)
    if not match:
        return None, None
    return _parse_decimal(match.group(1)), int(match.group(2))


def _parse_decimal(value):
    cleaned = value.strip()
    negative = cleaned.startswith("(") and cleaned.endswith(")")
    cleaned = cleaned.strip("()").replace(",", "")
    if not cleaned:
        return None
    try:
        number = Decimal(cleaned)
    except InvalidOperation:
        return None
    return -number if negative else number


def _to_int(value):
    if value is None:
        return None
    return int(value)
