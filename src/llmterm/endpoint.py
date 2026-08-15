"""OpenAI-compatible asynchronous LLM endpoint client.

Copyright (c) 2026 Prasanna Badami
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from openai import AsyncOpenAI


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
        return await self.client.chat.completions.create(**kwargs)

    async def close(self) -> None:
        """Close the underlying asynchronous HTTP client."""
        await self.client.close()
