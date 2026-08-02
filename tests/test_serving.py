import asyncio
import threading
from collections.abc import Iterator

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from pythonic.model import Decoder, DecoderConfig
from pythonic.serving import GenerationRequest, GenerationService, create_app


def model() -> Decoder:
    return Decoder(DecoderConfig(width=16, layers=1, context=16))


def test_sse_validation_and_health() -> None:
    with TestClient(create_app(model())) as client:
        assert client.get("/health").json() == {"ready": True}
        response = client.post("/generate", json={"prompt": "Trial", "max_new": 3})
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        assert "event: done" in response.text
        assert client.post("/generate", json={"prompt": "Trial", "max_new": 20}).status_code == 422
        assert (
            client.post("/generate", json={"prompt": "Trial", "temperature": -1}).status_code == 422
        )


def test_capacity_and_closed_stream_release_slots() -> None:
    async def run() -> None:
        service = GenerationService(model(), concurrency=1)
        slot = service.acquire()
        with pytest.raises(HTTPException) as error:
            service.acquire()
        assert error.value.status_code == 503

        async def connected() -> bool:
            return False

        stream = service.stream(GenerationRequest(prompt="Trial", max_new=3), slot, connected)
        await anext(stream)
        await stream.aclose()
        assert service.acquire() == slot

    asyncio.run(run())


def test_disconnected_client_stops_and_returns_capacity() -> None:
    async def run() -> None:
        service = GenerationService(model(), concurrency=1)
        slot = service.acquire()

        async def disconnected() -> bool:
            return True

        assert [
            chunk
            async for chunk in service.stream(GenerationRequest(prompt="Trial"), slot, disconnected)
        ] == []
        assert service.slots.qsize() == 1

    asyncio.run(run())


def test_cancelled_worker_keeps_slot_until_thread_finishes(monkeypatch: pytest.MonkeyPatch) -> None:
    started = threading.Event()
    release = threading.Event()

    def slow_tokens(*args: object, **kwargs: object) -> Iterator[int]:
        started.set()
        assert release.wait(timeout=2)
        yield 65

    monkeypatch.setattr("pythonic.serving.generate_tokens", slow_tokens)

    async def run() -> None:
        service = GenerationService(model(), concurrency=1)

        async def connected() -> bool:
            return False

        stream = service.stream(GenerationRequest(prompt="Trial"), service.acquire(), connected)
        task = asyncio.create_task(anext(stream))
        while not started.is_set():
            await asyncio.sleep(0.001)
        task.cancel()
        await asyncio.sleep(0.01)
        assert service.slots.empty()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert service.slots.qsize() == 1

    asyncio.run(run())
