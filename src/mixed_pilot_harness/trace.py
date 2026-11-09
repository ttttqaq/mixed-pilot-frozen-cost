"""Append-only JSONL recording with a standards-compatible SSE mirror."""

from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Any, Dict, Optional


class JsonlTrace:
    def __init__(self, path: Optional[Path], sse_path: Optional[Path] = None) -> None:
        self.path = path
        self.sse_path = sse_path
        self.sequence = 0
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            if self.sse_path is None:
                self.sse_path = path.with_suffix(".sse")
        if self.sse_path is not None:
            self.sse_path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, event: str, **data: Any) -> Dict[str, Any]:
        self.sequence += 1
        record = {
            "sequence": self.sequence,
            "event": event,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            **data,
        }
        if self.path is not None:
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        if self.sse_path is not None:
            with self.sse_path.open("a", encoding="utf-8") as stream:
                stream.write("id: %d\n" % self.sequence)
                stream.write("event: %s\n" % event)
                stream.write("data: %s\n\n" % json.dumps(record, ensure_ascii=False))
        return record
