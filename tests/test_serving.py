import asyncio

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
