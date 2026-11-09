"""Weighted profile selection, with injectable checkpointed RNG for inference."""
import math
import random
from .types import RouterDecision


class RandomRouterEngine:
    def __init__(self, config, rng=None):
        self.tiers = list(config.tiers)
        self.rng = rng if rng is not None else random.Random(config.seed)
        self.weighted = bool(config.possibility)
        if self.weighted:
            weights = [config.possibility[config.tiers[t].name] for t in self.tiers]
            scale = max(weights)
            weights = [w / scale for w in weights]
            total = math.fsum(weights)
            self.probabilities = {t:w/total for t,w in zip(self.tiers,weights)}
        else:
            self.probabilities = {t:1/len(self.tiers) for t in self.tiers}

    def route(self, request):
        tier = (self.rng.choices(self.tiers, weights=list(self.probabilities.values()), k=1)[0]
                if self.weighted else self.rng.choice(self.tiers))
        probability = self.probabilities[tier]
        return RouterDecision(tier=tier, route_class=tier, confidence=probability, source='random',
            probabilities=dict(self.probabilities),
            metadata={'api_step_index':request.api_step_index, 'decision_mode':'weighted_random' if self.weighted else 'uniform_random',
                      'selection_probability':probability})
