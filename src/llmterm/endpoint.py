"""OpenAI-compatible asynchronous LLM endpoint client.

Copyright (c) 2026 Prasanna Badami
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

from openai import APIError, AsyncOpenAI

# Some OpenAI-compatible servers send `data: [DONE]` but leave the HTTP body
# open (keep-alive comments, a trailing usage event, or a delayed chunked-EOF).
# The official SDK stops at [DONE] and acloses the response, which leaves
# httpcore2's PoolByteStream.__aiter__ suspended.  On Python 3.13, closing
# that generator raises: RuntimeError: generator didn't stop after athrow()
_STREAM_DRAIN_TIMEOUT_SECONDS = 1.0


class LLMEndpoint:
    """Configuration and async client for an OpenAI-compatible LLM endpoint."""

    def __init__(
        self,
        name: str,
        base_url: str,
        api_key: str | None = "dummy-key",
    ) -> None:
        self.name = name
        self.base_url = base_url
        self.api_key = api_key
        self.model: str | None = None
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            max_retries=3,
        )

    async def list_models(self) -> list[str]:
        """Return model IDs exposed by the endpoint."""
        response = await self.client.models.list()
        return [model.id for model in response.data]

    async def chat(self, **kwargs):
        """Create a chat completion using the configured endpoint."""
        result = await self.client.chat.completions.create(**kwargs)
        if kwargs.get("stream"):
            attach_stream_body_drain(result)
        return result

    async def close(self) -> None:
        """Close the underlying asynchronous HTTP client."""
        await self.client.close()


def attach_stream_body_drain(stream: Any) -> None:
    """Make a chat stream consume leftover HTTP body bytes before close.

    Replaces the SDK iterator so `data: [DONE]` does not abandon httpcore2's
    async generators.  The original unused ``__stream__()`` generator is
    closed on first iteration.
    """
    unused = getattr(stream, "_iterator", None)
    stream._iterator = _iter_and_drain(stream, unused)


async def _iter_and_drain(stream: Any, unused: Any) -> AsyncIterator[Any]:
    if unused is not None:
        await _aclose_quietly(unused)

    events = stream._iter_events()
    try:
        async for sse in events:
            if sse.data.startswith("[DONE]"):
                break
            chunk = _chunk_from_sse(stream, sse)
            if chunk is not None:
                yield chunk
    finally:
        await _drain_async_iterator(events)
        try:
            await stream.response.aclose()
        except Exception:
            pass


def _chunk_from_sse(stream: Any, sse: Any) -> Any | None:
    if not sse.data:
        return None

    try:
        data = sse.json()
    except (json.JSONDecodeError, ValueError):
        return None

    if isinstance(data, dict) and data.get("error"):
        error = data["error"]
        message = error.get("message") if isinstance(error, dict) else None
        if not isinstance(message, str) or not message:
            message = "An error occurred during streaming"
        raise APIError(
            message=message,
            request=stream.response.request,
            body=data["error"],
        )

    return stream._client._process_response_data(
        data=data,
        cast_to=stream._cast_to,
        response=stream.response,
    )


async def _drain_async_iterator(iterator: Any) -> None:
    """Finish an async iterator so nested httpcore generators exit cleanly."""

    async def consume() -> None:
        async for _ in iterator:
            pass

    try:
        await asyncio.wait_for(consume(), timeout=_STREAM_DRAIN_TIMEOUT_SECONDS)
    except (TimeoutError, Exception):
        await _aclose_quietly(iterator)


async def _aclose_quietly(iterator: Any) -> None:
    aclose = getattr(iterator, "aclose", None)
    if aclose is None:
        return
    try:
        await aclose()
    except Exception:
        pass
