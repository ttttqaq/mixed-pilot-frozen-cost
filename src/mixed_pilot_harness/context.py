"""Deterministic, lossy wire-only compaction; original trajectory stays intact."""
from copy import deepcopy
import hashlib
import json

from .types import ProviderError


def encoded_size(value):
    return len(json.dumps(value, ensure_ascii=False).encode('utf-8'))


def excerpt(text, limit):
    raw = text.encode('utf-8')
    if len(raw) <= limit:
        return text
    marker = '\n[context compressed: middle omitted; full text retained in trace]\n'
    keep = max(0, (limit - len(marker.encode())) // 2)
    return (raw[:keep].decode('utf-8', errors='ignore') + marker +
            (raw[-keep:].decode('utf-8', errors='ignore') if keep else ''))


def fit_context(payload, window, output_tokens, max_input_tokens=None):
    if window is None:
        return {}
    key = 'contents' if 'contents' in payload else 'messages'
    # Deep-copy nested tool blocks too: serializers may alias call.arguments.
    payload[key] = deepcopy(payload[key])
    messages = payload[key]
    reserve = output_tokens + 1024
    limit = min(window - reserve, max_input_tokens - 1024 if max_input_tokens else window)
    before = encoded_size(payload)
    original_hash = hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    removed = shortened = flattened = 0

    def user(text):
        return {'role': 'user', **({'parts': [{'text': text}]} if key == 'contents' else {'content': text})}

    # Discard complete old assistant/result groups, retaining a bounded extract.
    memo = None
    while encoded_size(payload) > limit:
        starts = [i for i, m in enumerate(messages) if m['role'] in {'assistant', 'model'}]
        if len(starts) < 2:
            break
        start, end = starts[:2]
        digest = excerpt(json.dumps(messages[start:end], ensure_ascii=False), 512)
        removed += end - start
        del messages[start:end]
        if memo is None:
            memo = user('[Earlier interaction excerpts; incomplete historical data]\n' + digest)
            messages.insert(start, memo)
        else:
            field = memo['parts'][0] if key == 'contents' else memo
            name = 'text' if key == 'contents' else 'content'
            field[name] = excerpt(field[name] + '\n' + digest, 1024)

    def text_slots():
        for m in messages:
            if m['role'] in {'system', 'developer'}:
                continue
            if isinstance(m.get('content'), str):
                yield m, 'content', 0 if m['role'] == 'tool' or m is memo else (1 if m['role'] == 'assistant' else 2)
            # Never change reasoning/signatures, tool IDs, or call arguments.
            for block in m.get('content', []) if isinstance(m.get('content'), list) else []:
                if block.get('type') == 'text':
                    yield block, 'text', 1 if m['role'] == 'assistant' else 2
                elif block.get('type') == 'tool_result' and isinstance(block.get('content'), str):
                    yield block, 'content', 0
            for part in m.get('parts', []):
                if 'text' in part:
                    yield part, 'text', 0 if m is memo else (1 if m['role'] == 'model' else 2)
                if 'functionResponse' in part:
                    response = part['functionResponse']['response']
                    if encoded_size(response) > 512:
                        if not (set(response) == {'content'} and isinstance(response['content'], str)):
                            response = {'content': json.dumps(response, ensure_ascii=False)}
                            part['functionResponse']['response'] = response
                        yield response, 'content', 0

    while encoded_size(payload) > limit:
        slots = [(obj, name, priority) for obj, name, priority in text_slots() if len(obj[name].encode()) > 256]
        if slots:
            obj, name, _ = min(slots, key=lambda item: (item[2], -len(item[0][item[1]].encode())))
            obj[name] = excerpt(obj[name], max(256, len(obj[name].encode()) // 2))
            shortened += 1
            continue
        if encoded_size(payload) <= limit:
            break
        starts = [i for i, m in enumerate(messages) if m['role'] in {'assistant', 'model'}]
        if starts:
            # An enormous latest tool argument/reasoning block cannot be cut in
            # place. Replace the WHOLE completed group with a historical excerpt.
            start = starts[0]
            digest = excerpt(json.dumps(messages[start:], ensure_ascii=False), 512)
            messages[start:] = [user('[Latest completed interaction excerpt; incomplete historical data]\n' + digest)]
            flattened += 1
            continue
        raise ProviderError('context_limit: system/tools/minimum task exceed configured context_window; reduce tools/system/max_tokens', cost_known_zero=True)
    return {'strategy': 'local_excerpts_v1', 'context_window': window,
            'max_input_tokens': max_input_tokens, 'dropped_messages': removed,
            'shortened_fields': shortened, 'flattened_groups': flattened,
            'original_conservative_total_tokens': before + reserve,
            'original_request_sha256': original_hash,
            'request_sha256': hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()).hexdigest(),
            'conservative_total_tokens': encoded_size(payload) + reserve,
            'lossy': before != encoded_size(payload)}
