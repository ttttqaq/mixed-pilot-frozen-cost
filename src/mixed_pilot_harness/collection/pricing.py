"""Explicit, versioned token price estimates; missing data is never free."""

from __future__ import annotations

import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

from ..types import Usage


def load_prices(path: Path, profiles: list[str]) -> dict:
    data = json.loads(path.read_text())
    if not data.get("currency") or not data.get("as_of"):
        raise ValueError("pricing requires currency and as_of")
    for profile in profiles:
        rates = data.get("profiles", {}).get(profile)
        if not rates:
            raise ValueError("missing pricing for " + profile)
        if rates.get("schedule", "fixed") not in {"fixed", "deepseek_cn_peak"}:
            raise ValueError("unsupported pricing schedule: " + profile)
        bands = rates.get("bands", [rates])
        if not bands:
            raise ValueError("empty price bands: " + profile)
        for band in bands:
            for key in ("input_per_million", "output_per_million"):
                value = band.get(key)
                if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                    raise ValueError("invalid %s for %s" % (key, profile))
            cached = band.get("cached_input_per_million", band["input_per_million"])
            if type(cached) not in (int, float) or not math.isfinite(cached) or cached < 0:
                raise ValueError("invalid cache price: " + profile)
    return data


def billing_terms(pricing: dict, profile: str) -> dict:
    """Compare effective rates, excluding snapshot dates and source annotations."""
    rates = pricing.get("profiles", {}).get(profile)
    if not rates:
        raise ValueError("missing pricing for " + profile)
    return {
        "currency": pricing.get("currency"),
        "schedule": rates.get("schedule", "fixed"),
        "bands": [
            {"max_input_tokens": band.get("max_input_tokens", float("inf")),
             "input_per_million": band["input_per_million"],
             "cached_input_per_million": band.get("cached_input_per_million", band["input_per_million"]),
             "output_per_million": band["output_per_million"]}
            for band in rates.get("bands", [rates])
        ],
    }


def estimate_cost(usage: Usage, rates: dict, timestamp: str) -> tuple[float, dict]:
    if usage.input_tokens < 0 or usage.output_tokens < 0 or usage.input_tokens + usage.output_tokens <= 0:
        raise ValueError("missing/invalid token usage; cannot estimate billed cost")
    bands = rates.get("bands", [rates])
    band = next((b for b in bands if usage.input_tokens <= b.get("max_input_tokens", float("inf"))), None)
    if band is None:
        raise ValueError("prompt exceeds configured price bands")
    raw = usage.raw
    if "prompt_cache_hit_tokens" in raw:
        cached = int(raw["prompt_cache_hit_tokens"])
    else:
        cached = int((raw.get("prompt_tokens_details") or {}).get("cached_tokens", 0))
    if not 0 <= cached <= usage.input_tokens:
        raise ValueError("invalid cached input token count")
    multiplier = 1.0
    if rates.get("schedule") == "deepseek_cn_peak":
        local = datetime.fromisoformat(timestamp).astimezone(timezone(timedelta(hours=8)))
        peak = local.weekday() < 5 and (9 <= local.hour < 12 or 14 <= local.hour < 18)
        multiplier = 1.0 if peak else 0.5
    price_in, price_out = band["input_per_million"], band["output_per_million"]
    price_cache = band.get("cached_input_per_million", price_in)
    cost = ((usage.input_tokens - cached) * price_in + cached * price_cache
            + usage.output_tokens * price_out) * multiplier / 1_000_000
    return cost, {"cached_input_tokens": cached, "time_multiplier": multiplier,
                  "rate_band": band, "basis": "token_estimate"}
