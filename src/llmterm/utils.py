"""Utility functions for llmterm.

Copyright (c) 2026 Prasanna Badami
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import os
import re
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant in general knowledge, physics, maths, "
    "science, engineering and technology."
)


def load_configuration() -> None:
    """Load environment variables from a .env file when present."""
    load_dotenv()


def get_unsloth_api_key() -> str:
    """Return the Unsloth Studio API key or a placeholder."""
    return os.environ.get("UNSLOTH_STUDIO_API_KEY", "dummy-key")


def clean_response(text: str) -> str:
    """Normalize common escaped Markdown/LaTeX delimiters."""
    text = text.replace(r"\[", "$$").replace(r"\]", "$$")
    text = text.replace(r"\(", "$").replace(r"\)", "$")
    return text


def set_system_prompt() -> str:
    """Ask whether to override the default system prompt."""
    choice = (
        input("Set a custom system prompt for this session? [y/N]: ").strip().lower()
    )

    if choice in {"y", "yes"}:
        return input(
            "Enter a system prompt for this session (press Enter to keep the default): "
        ).strip()

    return ""


def save_response_to_markdown(
    selected_model: str,
    prompt: str,
    response_text: str,
) -> None:
    """Append a prompt/response entry to ~/.llmterm/markdown-outputs/."""

    output_dir = Path.home() / ".llmterm" / "markdown-outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    sanitized_model = re.sub(r'[:/\\<>:"|?*]', "-", selected_model)
    date_str = datetime.now().strftime("%m-%d-%Y")
    filename = f"{sanitized_model}-{date_str}.md"
    timestamp = datetime.now().strftime("%H:%M:%S")

    with (output_dir / filename).open("a", encoding="utf-8") as handle:
        handle.write(f"# Prompt: {prompt}\n\n")
        handle.write(f"**Entry at {timestamp}**\n\n")
        handle.write(f"{response_text}\n\n")
        handle.write("---\n\n")
