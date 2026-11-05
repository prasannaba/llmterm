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

from llmterm.endpoints import is_valid_base_url, normalize_base_url

# Some OpenAI-compatible servers send `data: [DONE]` but leave the HTTP body
# open (keep-alive comments, a trailing usage event, or a delayed chunked-EOF).
# The official SDK stops at [DONE] and acloses the response, which can leave
# httpcore/httpx async generators suspended. Draining the leftover body first
# avoids `RuntimeError: generator didn't stop after athrow()` on cleanup.
_STREAM_DRAIN_TIMEOUT_SECONDS = 1.0


class LLMEndpoint:
    """Configuration and async client for an OpenAI-compatible LLM endpoint."""

    def __init__(
        self,
        name: str,
        base_url: str,
        api_key: str | None = None,
        *,
        timeout: float = 60.0,
        max_retries: int = 2,
    ) -> None:
        if not name or not name.strip():
            raise ValueError("Endpoint name must not be empty.")
        if not base_url or not is_valid_base_url(base_url):
            raise ValueError(f"Invalid base URL: {base_url!r}. Use http(s)://host[:port][/v1].")
        self.name = name.strip()
        self.base_url = normalize_base_url(base_url)
        # API key is held in memory only for the life of this object and is
        # never written to disk or logs by this class.
        self.api_key = api_key
        self.model: str | None = None
        self.client = AsyncOpenAI(
            api_key=api_key or "not-needed",
            base_url=self.base_url,
            timeout=timeout,
            max_retries=max_retries,
        )

    async def list_models(self) -> list[str]:
        """Return model IDs exposed by the endpoint."""
        response = await self.client.models.list()
        data = getattr(response, "data", None)
        if not isinstance(data, list):
            raise APIError(
                message="Unexpected model list response from server.",
                request=getattr(response, "request", None),
                body=None,
            )
        models: list[str] = []
        for item in data:
            model_id = getattr(item, "id", None)
            if isinstance(model_id, str) and model_id.strip():
                models.append(model_id)
        return models

    async def chat(self, **kwargs):
        """Create a chat completion using the configured endpoint."""
        if not kwargs.get("model"):
            raise ValueError("A model must be selected before chatting.")
        result = await self.client.chat.completions.create(**kwargs)
        if kwargs.get("stream"):
            attach_stream_body_drain(result)
        return result

    async def close(self) -> None:
        """Close the underlying HTTP client and drop the in-memory key."""
        try:
            await self.client.close()
        finally:
            # Best-effort memory hygiene: the key is never persisted, and the
            # reference is dropped here. (The SDK client object itself is also
            # discarded with this endpoint.)
            self.api_key = None


def _supports_drain_patch(stream: Any) -> bool:
    """Return True when the SDK stream exposes the expected internals."""
    return (
        hasattr(stream, "_iterator")
        and hasattr(stream, "_iter_events")
        and hasattr(stream, "_client")
        and hasattr(stream, "_cast_to")
        and hasattr(stream, "response")
    )


def attach_stream_body_drain(stream: Any) -> None:
    """Make a chat stream consume leftover HTTP body bytes before close.

    Replaces the SDK iterator so `data: [DONE]` does not abandon httpcore2's
    async generators. The original unused ``__stream__()`` generator is
    closed on first iteration. If the installed SDK no longer exposes the
    expected internals, this is a no-op (public iteration is used as-is).
    """
    if not _supports_drain_patch(stream):
        return
    unused = getattr(stream, "_iterator", None)
    stream._iterator = _iter_and_drain(stream, unused)


async def _iter_and_drain(stream: Any, unused: Any) -> AsyncIterator[Any]:
    if unused is not None:
        await _aclose_quietly(unused)

    events = stream._iter_events()
    try:
        async for sse in events:
            data = getattr(sse, "data", None)
            if isinstance(data, bytes):
                try:
                    data = data.decode("utf-8", errors="replace")
                except Exception:  # noqa: BLE001, S112 - skip undecodable keep-alive bytes
                    continue
            if not isinstance(data, str):
                continue
            if data.strip().startswith("[DONE]"):
                break
            chunk = _chunk_from_sse(stream, sse)
            if chunk is not None:
                yield chunk
    finally:
        await _drain_async_iterator(events)
        try:
            await stream.response.aclose()
        except Exception:  # noqa: BLE001, S110 - best-effort cleanup at teardown
            pass


def _chunk_from_sse(stream: Any, sse: Any) -> Any | None:
    data_attr = getattr(sse, "data", None)
    if isinstance(data_attr, bytes):
        try:
            data_attr = data_attr.decode("utf-8", errors="replace")
        except Exception:  # noqa: BLE001 - undecodable payload means no chunk
            return None
    if not isinstance(data_attr, str) or not data_attr.strip():
        return None

    try:
        data = sse.json()
    except (json.JSONDecodeError, ValueError, AttributeError):
        return None

    if isinstance(data, dict) and data.get("error"):
        error = data["error"]
        message = error.get("message") if isinstance(error, dict) else None
        if not isinstance(message, str) or not message:
            message = "An error occurred during streaming"
        raise APIError(
            message=message,
            request=getattr(getattr(stream, "response", None), "request", None),
            body=data["error"],
        )

    try:
        process = stream._client._process_response_data
    except AttributeError:
        return None
    return process(
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
        if hasattr(asyncio, "timeout"):
            async with asyncio.timeout(_STREAM_DRAIN_TIMEOUT_SECONDS):
                await consume()
        else:  # pragma: no cover - Python < 3.11 fallback
            await asyncio.wait_for(consume(), timeout=_STREAM_DRAIN_TIMEOUT_SECONDS)
    except TimeoutError:
        await _aclose_quietly(iterator)
    except Exception:  # noqa: BLE001 - drain is best-effort; always close quietly
        await _aclose_quietly(iterator)


async def _aclose_quietly(iterator: Any) -> None:
    aclose = getattr(iterator, "aclose", None)
    if aclose is None:
        return
    try:
        await aclose()
    except Exception:  # noqa: BLE001, S110 - quiet cleanup must never raise
        pass
