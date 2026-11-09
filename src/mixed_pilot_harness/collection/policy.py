"""Adapted BaselineMode/RouletteMode and propensity tracking from MTRouter.

See upstream/PROVENANCE.json and upstream/LICENSE.md. A private RNG makes
selections reproducible independently of task ordering and skipped runs.
"""

from __future__ import annotations

import random


class SamplingPolicy:
    def __init__(self, models: list[str], mode: str, seed: int, profile: str | None = None):
        if not models or len(set(models)) != len(models):
            raise ValueError("model pool must be nonempty with no duplicate profiles")
        if mode not in {"baseline", "roulette"}:
            raise ValueError("mode must be baseline or roulette")
        if mode == "baseline" and profile not in models:
            raise ValueError("baseline requires --profile from --model-pool")
        self.models, self.mode, self.profile = list(models), mode, profile
        self.rng = random.Random(seed)

    def choose(self) -> tuple[str, float]:
        if self.mode == "baseline":
            return self.profile, 1.0
        return self.rng.choice(self.models), 1.0 / len(self.models)
