import asyncio
import codecs
import json
from collections.abc import AsyncIterator, Awaitable, Callable

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from pythonic.generation import generate_tokens
from pythonic.model import ByteTokenizer, Decoder


class GenerationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    prompt: str = Field(min_length=1)
    max_new: int = Field(default=32, ge=1, le=4096)
    temperature: float = Field(default=0.0, ge=0)


def event(name: str, value: object) -> str:
    return f"event: {name}\ndata: {json.dumps(value)}\n\n"


class GenerationService:
    def __init__(self, model: Decoder, concurrency: int = 2) -> None:
        if concurrency <= 0:
            raise ValueError("Concurrency must be positive.")
        self.model = model.eval()
        self.slots: asyncio.Queue[int] = asyncio.Queue(maxsize=concurrency)
        for index in range(concurrency):
            self.slots.put_nowait(index)

    def acquire(self) -> int:
        try:
            return self.slots.get_nowait()
        except asyncio.QueueEmpty as error:
            raise HTTPException(503, "Generation capacity is full.") from error

    async def stream(
        self,
        body: GenerationRequest,
        slot: int,
        disconnected: Callable[[], Awaitable[bool]],
    ) -> AsyncIterator[str]:
        tokens = generate_tokens(
            self.model, ByteTokenizer.encode(body.prompt), body.max_new, body.temperature
        )
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        try:
            while not await disconnected():
                step = asyncio.create_task(asyncio.to_thread(next, tokens, None))
                try:
                    token = await asyncio.shield(step)
                except asyncio.CancelledError:
                    # A worker thread must finish before its capacity slot is reused.
                    await step
                    raise
                if token is None:
                    tail = decoder.decode(b"", final=True)
                    if tail:
                        yield event("text", {"text": tail})
                    yield event("done", {})
                    return
                text = decoder.decode(bytes([token])) if token < 256 else ""
                yield event("token", {"id": token, "text": text})
        finally:
            tokens.close()
            self.slots.put_nowait(slot)


def create_app(model: Decoder, concurrency: int = 2) -> FastAPI:
    app = FastAPI()
    service = GenerationService(model, concurrency)

    @app.get("/health")
    async def health() -> dict[str, bool]:
        return {"ready": True}

    @app.post("/generate")
    async def generate(body: GenerationRequest, request: Request) -> StreamingResponse:
        if len(ByteTokenizer.encode(body.prompt)) + body.max_new > model.config.context:
            raise HTTPException(422, "Prompt and output exceed model context.")
        slot = service.acquire()
        return StreamingResponse(
            service.stream(body, slot, request.is_disconnected),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache"},
        )

    return app
