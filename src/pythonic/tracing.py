import json
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class TraceWriter:
    def __init__(self, path: Path, metadata: dict[str, Any] | None = None) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.trace_id = uuid.uuid4().hex
        self.metadata = metadata or {}

    @contextmanager
    def span(self, name: str, attributes: dict[str, Any] | None = None) -> Iterator[dict[str, Any]]:
        values = dict(attributes or {})
        started = datetime.now(UTC).isoformat()
        begin = time.perf_counter()
        try:
            yield values
        except Exception as error:
            values["error"] = f"{type(error).__name__}: {error}"
            raise
        finally:
            record = {
                "trace_id": self.trace_id,
                "span_id": uuid.uuid4().hex,
                "name": name,
                "started": started,
                "duration_ms": (time.perf_counter() - begin) * 1000,
                "metadata": self.metadata,
                "attributes": values,
            }
            with self.path.open("a") as output:
                output.write(json.dumps(record, allow_nan=False) + "\n")
