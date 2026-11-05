"""Interactive command-line application for local LLM endpoints.

Copyright (c) 2026 Prasanna Badami
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections.abc import Sequence
from typing import Any

from openai import APIConnectionError, APIStatusError
from openai.types.chat import ChatCompletionMessageParam

from llmterm import __version__
from llmterm.endpoint import LLMEndpoint
from llmterm.endpoints import (
    CUSTOM_SENTINEL_KEY,
    KEY_PRIVACY_NOTICE,
    EndpointSpec,
    env_overrides,
    env_template,
    find_spec,
    is_valid_base_url,
    list_all_specs,
    normalize_base_url,
    save_custom_spec,
)
from llmterm.utils import (
    DEFAULT_SYSTEM_PROMPT,
    clean_response,
    get_api_key,
    load_configuration,
    save_response_to_markdown,
    set_system_prompt,
)

EXIT_COMMANDS = {"exit", "quit", "q"}


def parse_selection_index(raw: str, count: int) -> int | None:
    """Parse a 1-based menu choice into a 0-based index, else None."""
    text = raw.strip()
    if not text.isdigit():
        return None
    index = int(text) - 1
    if index < 0 or index >= count:
        return None
    return index


def build_endpoint(
    spec: EndpointSpec,
    *,
    base_url_override: str | None = None,
    api_key_env_override: str | None = None,
    prompt_for_key: bool = True,
) -> LLMEndpoint | None:
    """Create a client for the selected spec (single lazy client).

    Resolves the API key in memory only. Returns None when a required key
    is unavailable; callers should abort with guidance (already printed).
    """
    env_over = env_overrides()
    base_url = spec.resolved_base_url(base_url_override or env_over.get("base_url"))
    if spec.key == CUSTOM_SENTINEL_KEY and base_url_override:
        base_url = normalize_base_url(base_url_override)
    if not base_url:
        print(
            f"No base URL configured for {spec.name}. Provide --endpoint-url or choose Custom.",
            file=sys.stderr,
        )
        return None
    if not is_valid_base_url(base_url):
        print(f"Invalid base URL: {base_url!r}.", file=sys.stderr)
        return None

    key_env = api_key_env_override.strip() if api_key_env_override else spec.key_env
    if spec.requires_auth or key_env:
        key = get_api_key(
            env_var=key_env,
            endpoint_name=spec.name,
            required=spec.requires_auth,
            prompt=prompt_for_key,
        )
        if spec.requires_auth and not key:
            return None
    else:
        key = None
    return LLMEndpoint(spec.name, base_url, key)


def choose_endpoint_interactive(
    specs: list[EndpointSpec],
) -> tuple[EndpointSpec | None, str | None]:
    """Display endpoints and return (spec, base_url_override)."""
    print("Available LLM servers:")
    for index, spec in enumerate(specs, start=1):
        tag = " (needs API key, never stored)" if spec.requires_auth else ""
        print(f"{index}. {spec.name}{tag}")
    print(f"{len(specs) + 1}. Custom (enter OpenAI-compatible base URL)")

    for _ in range(3):
        try:
            raw = input("Select the endpoint: ")
        except (EOFError, KeyboardInterrupt):
            print("\nSelection cancelled.", file=sys.stderr)
            return None, None
        parsed = parse_selection_index(raw, len(specs) + 1)
        if parsed is None:
            print(
                f"Invalid selection {raw.strip()!r}. Enter a number 1-{len(specs) + 1}.",
                file=sys.stderr,
            )
            continue
        if parsed == len(specs):
            return handle_custom_endpoint()
        return specs[parsed], None
    print("Too many invalid attempts. Exiting.", file=sys.stderr)
    return None, None


def handle_custom_endpoint() -> tuple[EndpointSpec | None, str | None]:
    """Prompt for an ad-hoc base URL; optionally persist the URL (never keys)."""
    try:
        base_url = input("Enter OpenAI-compatible base URL (e.g. https://host/v1): ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nSelection cancelled.", file=sys.stderr)
        return None, None
    if not base_url or not is_valid_base_url(base_url):
        print(f"Invalid base URL: {base_url!r}.", file=sys.stderr)
        return None, None
    print(KEY_PRIVACY_NOTICE, file=sys.stderr)
    try:
        save = input("Remember this URL (not any key) for next time? [y/N]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        save = "n"
    if save in {"y", "yes"}:
        try:
            name = input("Name for this endpoint [Custom]: ").strip() or "Custom"
        except (EOFError, KeyboardInterrupt):
            print()
            name = "Custom"
        try:
            key_env = input("Env var holding its API key [optional]: ").strip() or None
        except (EOFError, KeyboardInterrupt):
            print()
            key_env = None
        try:
            spec = save_custom_spec(name, base_url, key_env)
            print(f"Saved URL for {spec.name} (keys are never saved).")
            return spec, None
        except ValueError as exc:
            print(f"Not saving custom endpoint: {exc}", file=sys.stderr)
    custom = EndpointSpec(
        key=CUSTOM_SENTINEL_KEY,
        name="Custom",
        base_url=normalize_base_url(base_url),
        key_env=None,
        requires_auth=True,
        help="Ad-hoc OpenAI-compatible endpoint.",
    )
    return custom, normalize_base_url(base_url)


async def choose_model(endpoint: LLMEndpoint, preset: str | None = None) -> bool:
    """Fetch models and let the user select one."""
    try:
        print("Fetching available models...")
        models = await endpoint.list_models()
    except Exception as exc:  # noqa: BLE001 - any discovery failure is user-facing
        print(f"\n[Error] Could not retrieve models: {exc}", file=sys.stderr)
        return False

    if not models:
        print("No models found on this endpoint.", file=sys.stderr)
        return False

    if preset:
        if preset in models:
            endpoint.model = preset
            print(f"Selected model: {endpoint.model}")
            return True
        print(
            f"Model {preset!r} not found on {endpoint.name}. Available: {', '.join(models[:10])}",
            file=sys.stderr,
        )
        return False

    print("Available LLMs:")
    for index, model in enumerate(models, start=1):
        print(f"{index}. {model}")

    for _ in range(3):
        try:
            raw = input("Select the LLM: ")
        except (EOFError, KeyboardInterrupt):
            print("\nSelection cancelled.", file=sys.stderr)
            return False
        parsed = parse_selection_index(raw, len(models))
        if parsed is None:
            print(
                f"Invalid selection {raw.strip()!r}. Enter 1-{len(models)}.",
                file=sys.stderr,
            )
            continue
        endpoint.model = models[parsed]
        print(f"Selected model: {endpoint.model}")
        return True
    print("Too many invalid attempts. Exiting.", file=sys.stderr)
    return False


async def _consume_stream(stream) -> str:
    """Print and accumulate a streaming chat completion."""
    full_content = ""
    print("Assistant: ", end="", flush=True)
    async for chunk in stream:
        if not chunk.choices:
            continue
        content = chunk.choices[0].delta.content
        if content is not None:
            print(content, end="", flush=True)
            full_content += content
    return full_content


async def run_conversation(
    endpoint: LLMEndpoint,
    *,
    save_to_markdown: bool | None = None,
    system_prompt_arg: str | None = None,
) -> None:
    """Run the interactive streaming conversation."""
    if save_to_markdown is None:
        try:
            save_choice = input("Save session to markdown? [y/N]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            save_choice = "n"
        save_to_markdown = save_choice in {"y", "yes"}

    system_prompt = system_prompt_arg or set_system_prompt() or DEFAULT_SYSTEM_PROMPT
    print(f"System Prompt: {system_prompt}")
    print("Commands: exit/quit/q to quit, clear to reset history.")

    messages: list[ChatCompletionMessageParam] = [{"role": "system", "content": system_prompt}]

    while True:
        try:
            user_prompt = input("\nEnter prompt (or 'exit/quit/q' to quit): ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if user_prompt.lower() in EXIT_COMMANDS:
            break
        if user_prompt.lower() == "clear":
            messages = [{"role": "system", "content": system_prompt}]
            print("Conversation history cleared.")
            continue
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
            if hasattr(stream, "__aenter__"):
                async with stream:
                    full_content = await _consume_stream(stream)
            else:  # pragma: no cover - defensive for SDK shape changes
                try:
                    full_content = await _consume_stream(stream)
                finally:
                    aclose = getattr(stream, "aclose", None) or getattr(stream, "close", None)
                    if callable(aclose):
                        result = aclose()
                        if asyncio.iscoroutine(result):
                            await result

            print()
            if not full_content.strip():
                messages.pop()  # roll back: don't pollute history with empties
                print("[Warning] Empty response; not added to history.", file=sys.stderr)
                continue
            messages.append({"role": "assistant", "content": full_content})

            if save_to_markdown:
                try:
                    path = save_response_to_markdown(
                        endpoint.model or "unknown-model",
                        user_prompt,
                        clean_response(full_content),
                    )
                    print(f"(saved to {path})")
                except ValueError:
                    messages.pop()
                    print("[Warning] Empty response; nothing saved.", file=sys.stderr)
                except OSError as exc:
                    print(f"[Error] Could not save markdown: {exc}", file=sys.stderr)

        except APIConnectionError as exc:
            messages.pop()  # roll back failed turn so retry isn't duplicated
            print(f"\n[Error] Lost connection to the LLM server: {exc}", file=sys.stderr)
        except APIStatusError as exc:
            messages.pop()
            print(
                f"\n[Error] API returned status {exc.status_code}: {exc}",
                file=sys.stderr,
            )
        except Exception as exc:  # noqa: BLE001 - any stream failure is user-facing
            messages.pop()
            print(f"\n[Error] Streaming failed: {exc}", file=sys.stderr)


def _suppress_httpcore_asyncgen_close_errors(
    loop: asyncio.AbstractEventLoop,
) -> None:
    """Hide a known Python 3.13 + httpcore2 cleanup error.

    When a streaming response is closed while PoolByteStream.__aiter__ is
    still suspended, httpcore2's safe_async_iterate() raises
    RuntimeError: generator didn't stop after athrow().  asyncio then
    reports that from shutdown_asyncgens().  The stream drain in
    endpoint.py avoids this in the common case; this handler is the
    fallback for servers that never finish the HTTP body. It matches the
    exact cleanup message and delegates everything else.
    """
    previous = loop.get_exception_handler()

    def handler(event_loop: asyncio.AbstractEventLoop, context: dict[str, Any]) -> None:
        message = str(context.get("message", ""))
        exc = context.get("exception")
        if (
            isinstance(exc, RuntimeError)
            and "generator didn't stop after athrow" in str(exc)
            and ("asynchronous generator" in message or context.get("asyncgen") is not None)
        ):
            return
        if previous is not None:
            previous(event_loop, context)
        else:
            event_loop.default_exception_handler(context)

    loop.set_exception_handler(handler)


async def async_main(args: argparse.Namespace) -> int:
    """Application entry point."""
    load_configuration()
    _suppress_httpcore_asyncgen_close_errors(asyncio.get_running_loop())
    print(f"llmterm {__version__} - {KEY_PRIVACY_NOTICE}", file=sys.stderr)

    specs = list_all_specs()
    if args.list_endpoints:
        for spec in specs:
            auth = f" [key: {spec.key_env}]" if spec.key_env else ""
            print(f"{spec.name}{auth}: {spec.base_url or '(URL required)'}")
        return 0

    if args.endpoint:
        spec = find_spec(args.endpoint)
        if spec is None:
            print(
                f"Unknown endpoint {args.endpoint!r}. Use --list-endpoints.",
                file=sys.stderr,
            )
            return 2
        base_override = args.endpoint_url
    else:
        if not sys.stdin.isatty():
            print(
                "No --endpoint given and input is not interactive. "
                "Use --endpoint/--list-endpoints.",
                file=sys.stderr,
            )
            return 2
        spec, base_override = choose_endpoint_interactive(specs)
        if spec is None:
            return 1
        if args.endpoint_url:
            base_override = args.endpoint_url

    endpoint = build_endpoint(
        spec,
        base_url_override=base_override,
        api_key_env_override=args.api_key_env,
        prompt_for_key=sys.stdin.isatty() and not args.no_prompt,
    )
    if endpoint is None:
        return 1

    print(f"Selected LLM Endpoint: {endpoint.name} ({endpoint.base_url})")

    try:
        if args.list_models:
            try:
                for model in await endpoint.list_models():
                    print(model)
                return 0
            except Exception as exc:  # noqa: BLE001 - any discovery failure exits 1
                print(f"[Error] Could not retrieve models: {exc}", file=sys.stderr)
                return 1
        if await choose_model(endpoint, preset=args.model):
            await run_conversation(
                endpoint,
                save_to_markdown=False if args.no_save else None,
                system_prompt_arg=args.system_prompt,
            )
        else:
            return 1
        return 0
    finally:
        # asyncio.run() normally shuts down async generators only after this
        # coroutine returns.  Drain them first so httpcore2 finishes response
        # cleanup while its client/connection pool is still available.
        await asyncio.get_running_loop().shutdown_asyncgens()
        await endpoint.close()


def create_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser."""
    parser = argparse.ArgumentParser(
        prog="llmterm",
        description=(
            "Interactive terminal client for OpenAI-compatible LLM servers. "
            "Select an endpoint and model, then start a streaming conversation. "
            f"{KEY_PRIVACY_NOTICE}"
        ),
        epilog=(
            "Configured endpoints: Llama.cpp, LiteRTM, UnSloth-Studio, "
            "LM-Studio, Ollama, OpenAI, Meta-AI, Baseten, plus Custom URLs. "
            "Use --list-endpoints to see all."
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    parser.add_argument(
        "--endpoint",
        help="Endpoint name or key (see --list-endpoints). Skips the menu.",
    )
    parser.add_argument(
        "--endpoint-url",
        help="Override the base URL (e.g. your Baseten deployment /v1 URL).",
    )
    parser.add_argument(
        "--api-key-env",
        help="Env var holding the API key (key itself is never accepted/stored).",
    )
    parser.add_argument("--model", help="Model ID to use (skips model menu).")
    parser.add_argument("--system-prompt", help="System prompt (skips the interactive prompt).")
    parser.add_argument("--list-endpoints", action="store_true", help="List endpoints and exit.")
    parser.add_argument("--list-models", action="store_true", help="List models and exit.")
    parser.add_argument("--no-save", action="store_true", help="Never write markdown logs.")
    parser.add_argument(
        "--no-prompt",
        action="store_true",
        help="Never prompt for keys (fail if env var missing). For scripts.",
    )
    parser.add_argument(
        "--show-env-template",
        action="store_true",
        help="Print a .env template (variable names only, no values) and exit. "
        "Useful when llmterm was installed via `uv tool` and .env.example "
        "is not on disk.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Synchronous console-script entry point."""
    parser = create_parser()
    args = parser.parse_args(argv)

    if args.show_env_template:
        print(env_template())
        raise SystemExit(0)

    try:
        code = asyncio.run(async_main(args))
    except (KeyboardInterrupt, EOFError):
        print("\nExiting.", file=sys.stderr)
        code = 130
    raise SystemExit(code)


if __name__ == "__main__":
    main()
