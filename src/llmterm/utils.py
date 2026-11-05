"""Utility functions for llmterm.

Copyright (c) 2026 Prasanna Badami
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import getpass
import os
import re
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from llmterm.endpoints import KEY_PRIVACY_NOTICE

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant in general knowledge, physics, maths, "
    "science, engineering and technology."
)

_WINDOWS_RESERVED_NAMES = {
    "con",
    "prn",
    "aux",
    "nul",
    *[f"com{i}" for i in range(1, 10)],
    *[f"lpt{i}" for i in range(1, 10)],
}


def load_configuration() -> None:
    """Load environment variables from a .env file when present.

    Read-only: this never writes keys anywhere. ``.env`` is resolved from
    the current working directory by python-dotenv.
    """
    load_dotenv()


def get_api_key(
    *,
    env_var: str | None,
    endpoint_name: str,
    required: bool,
    prompt: bool = True,
) -> str | None:
    """Resolve an API key from the environment or a secure prompt.

    The key is returned in memory only and is never written to disk, logged,
    or saved by the caller. Returns ``None`` when no key is needed/available.
    """
    if env_var:
        value = os.environ.get(env_var, "").strip()
        if value:
            return value
    if not required:
        return None
    if env_var and not os.environ.get(env_var, "").strip():
        print(
            f"No {env_var} found in the environment. {KEY_PRIVACY_NOTICE}",
            file=sys.stderr,
        )
    if not prompt or not sys.stdin.isatty():
        if env_var:
            print(
                f"Missing required API key: set {env_var} and retry. "
                "Keys are never stored by llmterm.",
                file=sys.stderr,
            )
        return None
    print(KEY_PRIVACY_NOTICE, file=sys.stderr)
    try:
        entered = getpass.getpass(
            f"Enter API key for {endpoint_name} (input hidden, never stored): "
        ).strip()
    except (EOFError, KeyboardInterrupt):
        print(file=sys.stderr)
        return None
    if not entered:
        print("No API key entered.", file=sys.stderr)
        return None
    return entered


def get_unsloth_api_key() -> str | None:
    """Return the Unsloth Studio API key when set, else None (never a dummy)."""
    return os.environ.get("UNSLOTH_STUDIO_API_KEY", "").strip() or None


def clean_response(text: str) -> str:
    """Normalize common escaped Markdown/LaTeX delimiters.

    Only rewrites backslash-escaped ``\\[``/``\\]``/``\\(``/``\\)`` pairs and
    leaves fenced code blocks untouched to avoid corrupting code samples.
    Idempotent: running it twice yields the same output.
    """
    parts = text.split("```")
    for i in range(0, len(parts), 2):  # outside code fences only
        segment = parts[i]
        segment = segment.replace(r"\[", "$$").replace(r"\]", "$$")
        segment = segment.replace(r"\(", "$").replace(r"\)", "$")
        parts[i] = segment
    return "```".join(parts)


def set_system_prompt() -> str:
    """Ask whether to override the default system prompt."""
    try:
        choice = input("Set a custom system prompt for this session? [y/N]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return ""

    if choice in {"y", "yes"}:
        try:
            custom = input(
                "Enter a system prompt for this session (press Enter to keep the default): "
            )
        except (EOFError, KeyboardInterrupt):
            print()
            return ""
        # Preserve intentional spacing; treat whitespace-only as "keep default".
        if not custom.strip():
            return ""
        return custom.strip()

    return ""


def sanitize_model_for_filename(selected_model: str) -> str:
    """Sanitize a model ID for use as a filename stem."""
    sanitized = re.sub(r'[:/\\<>:"|?*\x00-\x1f]', "-", selected_model.strip())
    sanitized = sanitized.strip(" .")
    if not sanitized:
        sanitized = "unknown-model"
    if sanitized.lower() in _WINDOWS_RESERVED_NAMES:
        sanitized = f"{sanitized}-model"
    # Keep filenames within filesystem limits.
    if len(sanitized) > 100:
        sanitized = sanitized[:100].rstrip(" .")
    # Neutralize parent-directory segments without losing readability.
    sanitized = sanitized.replace("..", "--")
    return sanitized or "unknown-model"


def _escape_markdown_text(text: str) -> str:
    """Escape a prefix that could otherwise break markdown structure."""
    lines = text.splitlines() or [""]
    escaped: list[str] = []
    for line in lines:
        if line.lstrip().startswith(("#", "---", "```")):
            escaped.append("\\" + line)
        else:
            escaped.append(line)
    return "\n".join(escaped)


def save_response_to_markdown(
    selected_model: str,
    prompt: str,
    response_text: str,
) -> Path:
    """Append a prompt/response entry to ~/.llmterm/markdown-outputs/.

    Returns the file path written. Raises OSError on failure; callers decide
    how to report it. Never writes API keys: only prompt/response text given.
    """
    if not response_text.strip():
        raise ValueError("Refusing to save an empty response.")

    output_dir = Path.home() / ".llmterm" / "markdown-outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    sanitized_model = sanitize_model_for_filename(selected_model)
    now = datetime.now().astimezone()
    date_str = now.strftime("%Y-%m-%d")
    filename = f"{sanitized_model}-{date_str}.md"
    timestamp = now.strftime("%H:%M:%S %Z")

    path = output_dir / filename
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"# Prompt: {_escape_markdown_text(prompt)}\n\n")
        handle.write(f"**Entry at {timestamp}**\n\n")
        handle.write(f"{response_text}\n\n")
        handle.write("---\n\n")
    return path
