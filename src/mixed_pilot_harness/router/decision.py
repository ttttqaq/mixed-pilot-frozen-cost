"""Select the final route without coupling model inference to policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Mapping

from .config import RouterDecisionConfig


ROUTE_CLASSES = ("R0", "R1", "R2", "R3")
_ROUTE_INDEX = {route: index for index, route in enumerate(ROUTE_CLASSES)}


@dataclass(frozen=True)
class RouteSelection:
    raw_route_class: str
    legacy_route_class: str
    selected_route_class: str
    applied_rules: List[str]


class RouteSelectionError(ValueError):
    """The inference result cannot be converted into a four-class route."""


def _raw_argmax(probabilities: Mapping[str, float]) -> str:
    missing = [route for route in ROUTE_CLASSES if route not in probabilities]
    if missing:
        raise RouteSelectionError(
            "router probabilities are missing classes: %s" % ", ".join(missing)
        )
    # ROUTE_CLASSES order makes ties deterministic and conservatively prefers
    # the lower class, matching numpy.argmax in the inherited V4 core.
    return max(ROUTE_CLASSES, key=lambda route: float(probabilities[route]))


def select_route(
    probabilities: Mapping[str, float],
    legacy_route_class: str,
    legacy_flags: Mapping[str, bool],
    config: RouterDecisionConfig,
) -> RouteSelection:
    """Choose a route using one of the adapter's three decision modes."""

    raw_route = _raw_argmax(probabilities)
    if legacy_route_class not in _ROUTE_INDEX:
        raise RouteSelectionError(
            "OpenSquilla returned unsupported route class %r" % legacy_route_class
        )

    if config.mode == "legacy":
        selected = legacy_route_class
        rules = ["opensquilla_legacy_postprocess"]
    elif config.mode == "raw":
        selected = raw_route
        rules = ["raw_argmax"]
    elif config.mode == "api_step":
        selected = raw_route
        rules = ["raw_argmax"]
        if (
            config.high_risk_floor
            and bool(legacy_flags.get("high_risk", False))
            and _ROUTE_INDEX[selected] < _ROUTE_INDEX["R2"]
        ):
            selected = "R2"
            rules.append("high_risk_floor:R2")
    else:  # RouterConfig validation normally makes this unreachable.
        raise RouteSelectionError("unsupported decision mode %r" % config.mode)

    return RouteSelection(
        raw_route_class=raw_route,
        legacy_route_class=legacy_route_class,
        selected_route_class=selected,
        applied_rules=rules,
    )
