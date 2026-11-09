"""Loopback HTTP scoring service; the harness keeps ownership of provider calls."""
from __future__ import annotations
import argparse
import json
import math
from http.server import BaseHTTPRequestHandler, HTTPServer


def score_request(scorer, payload):
    models = payload.get('models')
    if models != scorer.model_names:
        raise ValueError('models must match checkpoint names and order exactly')
    messages = payload.get('messages')
    if not isinstance(messages, list) or not messages:
        raise ValueError('messages must be a nonempty list')
    if any(not isinstance(m, dict) or m.get('role') not in {'system','user','assistant','tool'}
           or not isinstance(m.get('content'), str) for m in messages):
        raise ValueError('each message requires role and string content')
    values = [float(v) for v in scorer.predict(messages, models)]
    if len(values) != len(models) or not all(math.isfinite(v) for v in values):
        raise ValueError('checkpoint returned invalid Q-values')
    idx = max(range(len(models)), key=lambda i: values[i])
    return {'model': models[idx], 'q_values': dict(zip(models, values)),
            'model_names': models, 'selection': 'argmax'}


def make_server(scorer, host='127.0.0.1', port=8765):
    class Handler(BaseHTTPRequestHandler):
        def reply(self, status, value):
            raw = json.dumps(value, allow_nan=False).encode()
            self.send_response(status); self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(raw))); self.end_headers(); self.wfile.write(raw)
        def do_GET(self):
            if self.path != '/health': return self.reply(404, {'error':'not found'})
            self.reply(200, {'model_names':scorer.model_names, 'status':'ready'})
        def do_POST(self):
            if self.path != '/route': return self.reply(404, {'error':'not found'})
            try:
                n = int(self.headers.get('Content-Length', '0'))
                if not 0 < n <= 16*1024*1024: raise ValueError('invalid request size')
                payload = json.loads(self.rfile.read(n))
                if not isinstance(payload, dict): raise ValueError('expected JSON object')
                result = score_request(scorer, payload)
            except (ValueError, TypeError) as exc:
                return self.reply(400, {'error': str(exc)})
            except Exception as exc:
                return self.reply(500, {'error': str(exc)})
            self.reply(200, result)
        def setup(self):
            super().setup(); self.connection.settimeout(120)
        def log_message(self, fmt, *args): pass
    # Serialize requests: a single shared encoder/torch module is not re-entered.
    return HTTPServer((host,port), Handler)


def load_scorer(checkpoint, device='cpu', embedding_url=None, model_metadata=None, expected_models=None):
    """Shared loader for the HTTP service and in-process harness routing."""
    from .research.training.q_network import NeuralQFunction
    from .research.routers.learned import validate_model_names
    names, config, tradeoff = NeuralQFunction.load_config_from_checkpoint(checkpoint)
    validate_model_names(names, expected_models)
    if embedding_url:
        if config.history_encoder_config is None or config.history_encoder_config.backend != 'vllm':
            raise ValueError('embedding_url requires a vllm checkpoint')
        config.history_encoder_config.vllm_base_url = embedding_url
    if model_metadata:
        from .research.training.model_encoder import ModelEncoderConfig
        if config.model_encoder_config is None: config.model_encoder_config = ModelEncoderConfig()
        config.model_encoder_config.custom_models_path = str(model_metadata)
    scorer = NeuralQFunction(names, config=config, lambda_=tradeoff)
    scorer.load(checkpoint); scorer.to(device); scorer.eval()
    return scorer


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--port', type=int, default=8765)
    p.add_argument('--device', choices=['cpu','cuda','mps'], default='cpu')
    p.add_argument('--embedding-url', help='Override vLLM endpoint only, preserving encoder identity')
    p.add_argument('--model-metadata', help='Relocated custom_models.yaml (same features as training)')
    a = p.parse_args()
    scorer = load_scorer(a.checkpoint, a.device, a.embedding_url, a.model_metadata)
    server = make_server(scorer, port=a.port)
    print(f'MTRouter ready at http://127.0.0.1:{a.port}; models={scorer.model_names}', flush=True)
    try: server.serve_forever()
    except KeyboardInterrupt: pass
    finally: server.server_close()

if __name__ == '__main__': main()
