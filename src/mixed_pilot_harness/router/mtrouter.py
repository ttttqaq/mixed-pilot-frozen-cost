"""Stdlib HTTP bridge to the separately installed MTRouter scoring service."""
from __future__ import annotations
import json
import math
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .opensquilla_v4 import RouterRuntimeError
from .types import RouterDecision


class MTRouterEngine:
    def __init__(self, config):
        self.config = config
        self.models = list(config.model_names) or [p.model for p in config.tiers.values()]
        self.model_to_tier = dict(zip(self.models, config.tiers))
        self.scorer = None
        if config.checkpoint:
            try:
                from mixed_pilot_mtrouter.service import load_scorer
            except ImportError as exc:
                raise RouterRuntimeError('Local MTRouter requires mixed-pilot-mtrouter and its dependencies in this Python environment; install -e ./MTRouter or use service_url') from exc
            self.scorer = load_scorer(str(config.checkpoint), device=config.device,
                embedding_url=config.embedding_url, model_metadata=config.model_metadata,
                expected_models=self.models)

    def route(self, request):
        # Match the upstream encoder's role/content view of training messages.
        messages = ([{'role':'system', 'content':request.system}] if request.system else [])
        messages += [{'role':m.role, 'content':m.content} for m in request.messages]
        payload = {'messages':messages, 'models':self.models}
        if self.scorer is not None:
            from mixed_pilot_mtrouter.service import score_request
            return self.decision(score_request(self.scorer, payload), request)
        req = Request(self.config.service_url + '/route',
                      data=json.dumps(payload).encode(), headers={'Content-Type':'application/json'})
        try:
            with urlopen(req, timeout=self.config.service_timeout) as response:
                result = json.load(response)
        except HTTPError as exc:
            detail = exc.read(2048).decode('utf-8', errors='replace')
            raise RouterRuntimeError('MTRouter HTTP %s: %s' % (exc.code, detail)) from exc
        except (URLError, OSError, ValueError) as exc:
            raise RouterRuntimeError('MTRouter service unavailable or invalid response: %s' % exc) from exc
        return self.decision(result, request)

    def decision(self, result, request):
        if not isinstance(result, dict) or result.get('model_names') != self.models:
            raise RouterRuntimeError('MTRouter model names/order do not match the configured tiers')
        scores = result.get('q_values')
        if not isinstance(scores, dict) or set(scores) != set(self.models):
            raise RouterRuntimeError('MTRouter response must score every configured model')
        if any(isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v) for v in scores.values()):
            raise RouterRuntimeError('MTRouter returned non-finite or non-numeric Q-values')
        selected = max(self.models, key=lambda name: scores[name])
        if result.get('model') != selected:
            raise RouterRuntimeError('MTRouter selected model does not match Q-value argmax')
        ranked = sorted(scores.values(), reverse=True)
        return RouterDecision(
            tier=self.model_to_tier[selected], route_class='M%d' % self.models.index(selected),
            confidence=0.0, source='mtrouter',
            margin=ranked[0]-ranked[1] if len(ranked)>1 else 0.0,
            metadata={'api_step_index':request.api_step_index, 'decision_mode':'q_argmax',
                      'q_values':scores, 'selected_model':selected, 'applied_rules':['q_argmax'],
                      'selection_probability':1.0,
                      'confidence_available':False},
        )
