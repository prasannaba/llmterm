"""Interactive command-line application for local LLM endpoints.

Copyright (c) 2026 Prasanna Badami
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import argparse
import asyncio
from typing import Any

from openai import APIConnectionError, APIStatusError
from openai.types.chat import ChatCompletionMessageParam

from llmterm import __version__
from llmterm.endpoint import LLMEndpoint
from llmterm.utils import (
    DEFAULT_SYSTEM_PROMPT,
    clean_response,
    get_unsloth_api_key,
    load_configuration,
    save_response_to_markdown,
    set_system_prompt,
)


def create_endpoints() -> list[LLMEndpoint]:
    """Create the configured OpenAI-compatible local LLM endpoints."""
    return [
        LLMEndpoint("Llama.cpp", "http://localhost:8080/v1", "non-needed"),
        LLMEndpoint("Google-Litert", "http://localhost:9379/v1", "non-needed"),
        LLMEndpoint("UnSloth", "http://127.0.0.1:8888/v1", get_unsloth_api_key()),
        LLMEndpoint("LM-Studio", "http://127.0.0.1:1234/v1", "non-needed"),
        LLMEndpoint("Ollama", "http://localhost:11434/v1", "non-needed"),
    ]


def choose_endpoint(endpoints: list[LLMEndpoint]) -> LLMEndpoint | None:
    """Display endpoints and return the user's selection."""
    print("Available LLM servers:")
    for index, endpoint in enumerate(endpoints, start=1):
        print(f"{index}. {endpoint.name}")

    try:
        return endpoints[int(input("Select the endpoint: ")) - 1]
    except (ValueError, IndexError):
        print("Invalid selection. Exiting.")
        return None


async def choose_model(endpoint: LLMEndpoint) -> bool:
    """Fetch models and let the user select one."""
    try:
        print("Fetching available models...")
        models = await endpoint.list_models()
    except Exception as exc:
        print(f"\n[Error] Could not retrieve models: {exc}")
        return False

    if not models:
        print("No models found on this endpoint.")
        return False

    print("Available LLMs:")
    for index, model in enumerate(models, start=1):
        print(f"{index}. {model}")

    try:
        endpoint.model = models[int(input("Select the LLM: ")) - 1]
    except (ValueError, IndexError):
        print("Invalid LLM selection. Exiting.")
        return False

    print(f"Selected model: {endpoint.model}")
    return True


async def run_conversation(endpoint: LLMEndpoint) -> None:
    """Run the interactive streaming conversation."""
    save_choice = input("Save session to markdown? [y/N]: ").strip().lower()
    system_prompt = set_system_prompt() or DEFAULT_SYSTEM_PROMPT
    print(f"System Prompt: {system_prompt}")

    messages: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": system_prompt}
    ]

    while True:
        user_prompt = input("\nEnter prompt (or 'exit/quit/q' to quit): ").strip()

        if user_prompt.lower() in {"exit", "quit", "q"}:
            break
        if not user_prompt:
            continue

        messages.append({"role": "user", "content": user_prompt})

        try:
            stream = await endpoint.chat(
                model=endpoint.model,
                messages=messages,
                stream=True,
            )

            # Drain leftover HTTP body bytes (see attach_stream_body_drain)
            # and let AsyncStream close the response exactly once.
            async with stream:
                full_content = ""
                print("Assistant: ", end="", flush=True)

                async for chunk in stream:
                    if not chunk.choices:
                        continue
                    content = chunk.choices[0].delta.content
                    if content is not None:
                        print(content, end="", flush=True)
                        full_content += content

            print()
            messages.append({"role": "assistant", "content": full_content})

            if save_choice in {"y", "yes"}:
                save_response_to_markdown(
                    endpoint.model or "unknown-model",
                    user_prompt,
                    clean_response(full_content),
                )

        except APIConnectionError:
            print("\n[Error] Lost connection to the LLM server during generation.")
        except APIStatusError as exc:
            print(f"\n[Error] API returned status {exc.status_code}: {exc.message}")
        except Exception as exc:
            print(f"\n[Error] Streaming failed: {exc}")


def _suppress_httpcore_asyncgen_close_errors(
    loop: asyncio.AbstractEventLoop,
) -> None:
    """Hide a known Python 3.13 + httpcore2 cleanup error.

    When a streaming response is closed while PoolByteStream.__aiter__ is
    still suspended, httpcore2's safe_async_iterate() raises
    RuntimeError: generator didn't stop after athrow().  asyncio then
    reports that from shutdown_asyncgens().  The stream drain in
    endpoint.py avoids this in the common case; this handler is the
    fallback for servers that never finish the HTTP body.
    """
    previous = loop.get_exception_handler()

    def handler(event_loop: asyncio.AbstractEventLoop, context: dict[str, Any]) -> None:
        message = str(context.get("message", ""))
        exc = context.get("exception")
        if (
            isinstance(exc, RuntimeError)
            and "generator didn't stop after athrow" in str(exc)
            and (
                "asynchronous generator" in message
                or context.get("asyncgen") is not None
            )
        ):
            return
        if previous is not None:
            previous(event_loop, context)
        else:
            event_loop.default_exception_handler(context)

    loop.set_exception_handler(handler)


async def async_main() -> None:
    """Application entry point."""
    load_configuration()
    _suppress_httpcore_asyncgen_close_errors(asyncio.get_running_loop())
    endpoint = choose_endpoint(create_endpoints())

    if endpoint is None:
        return

    print(f"Selected LLM Endpoint: {endpoint.name}")

    try:
        if await choose_model(endpoint):
            await run_conversation(endpoint)
    finally:
        # asyncio.run() normally shuts down async generators only after this
        # coroutine returns.  Drain them first so httpcore2 finishes response
        # cleanup while its client/connection pool is still available.
        await asyncio.get_running_loop().shutdown_asyncgens()
        await endpoint.close()


def create_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser."""
    return argparse.ArgumentParser(
        prog="llmterm",
        description=(
            "Interactive terminal client for OpenAI-compatible LLM servers. "
            "Select an endpoint and model, then start a streaming conversation."
        ),
        epilog=(
            "Configured endpoints: Llama.cpp, Google-Litert, UnSloth, "
            "LM-Studio, and Ollama."
        ),
    )


def main() -> None:
    """Synchronous console-script entry point."""
    parser = create_parser()
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    parser.parse_args()

    try:
        asyncio.run(async_main())
    except (KeyboardInterrupt, EOFError):
        pass


if __name__ == "__main__":
    main()
