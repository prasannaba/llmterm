"""Endpoint registry for llmterm.

Copyright (c) 2026 Prasanna Badami
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

KEY_PRIVACY_NOTICE = (
    "API keys are kept in memory only and are never written to disk, "
    "logged, or saved. Set the documented environment variable to avoid "
    "re-typing, or enter the key when prompted."
)


@dataclass
class EndpointSpec:
    """Non-secret description of an OpenAI-compatible endpoint."""

    key: str
    name: str
    base_url: str | None
    key_env: str | None = None
    requires_auth: bool = False
    help: str = ""
    is_custom: bool = False

    def resolved_base_url(self, override: str | None = None) -> str | None:
        """Return the effective base URL, preferring an explicit override."""
        if override:
            return normalize_base_url(override)
        if self.base_url:
            return normalize_base_url(self.base_url)
        return None


# NOTE: Meta AI and Baseten default URLs vary by account/deployment and can
# change. Defaults below are overridable via --endpoint-url,
# LLMTERM_BASE_URL, or the interactive Custom endpoint. They are intentionally
# conservative; when in doubt, paste your provider's OpenAI-compatible base URL.
BUILTIN_SPECS: list[EndpointSpec] = [
    EndpointSpec(
        key="llamacpp",
        name="Llama.cpp",
        base_url="http://localhost:8080/v1",
        requires_auth=False,
        help="Local llama.cpp server (no API key needed).",
    ),
    EndpointSpec(
        key="litertm",
        name="LiteRTM",
        base_url="http://localhost:9379/v1",
        requires_auth=False,
        help="Local LiteRTM server (no API key needed).",
    ),
    EndpointSpec(
        key="lmstudio",
        name="LM-Studio",
        base_url="http://127.0.0.1:1234/v1",
        requires_auth=False,
        help="Local LM Studio server (no API key needed).",
    ),
    EndpointSpec(
        key="ollama",
        name="Ollama",
        base_url="http://localhost:11434/v1",
        requires_auth=False,
        help="Local Ollama server (no API key needed).",
    ),
    EndpointSpec(
        key="unsloth",
        name="UnSloth-Studio",
        base_url="http://127.0.0.1:8888/v1",
        key_env="UNSLOTH_STUDIO_API_KEY",
        requires_auth=False,
        help="UnSloth Studio server. Key optional; set UNSLOTH_STUDIO_API_KEY if enabled.",
    ),
    EndpointSpec(
        key="openai",
        name="OpenAI",
        base_url="https://api.openai.com/v1",
        key_env="OPENAI_API_KEY",
        requires_auth=True,
        help="Hosted OpenAI API. Requires OPENAI_API_KEY.",
    ),
    EndpointSpec(
        key="meta",
        name="Meta-AI",
        base_url="https://api.meta.ai/v1",
        key_env="META_API_KEY",
        requires_auth=True,
        help=(
            "Hosted Meta AI API (OpenAI-compatible). Requires META_API_KEY. "
            "Override the base URL if your account uses a different endpoint."
        ),
    ),
    EndpointSpec(
        key="baseten",
        name="Baseten",
        base_url="https://inference.baseten.co/v1",
        key_env="BASETEN_API_KEY",
        requires_auth=True,
        help=(
            "Hosted Baseten inference (OpenAI-compatible). Requires "
            "BASETEN_API_KEY. Override the base URL for a dedicated "
            "deployment's /v1 endpoint."
        ),
    ),
]

# Synthetic entry appended at selection time for ad-hoc URLs.
CUSTOM_SENTINEL_KEY = "__custom__"


def normalize_base_url(url: str) -> str:
    """Normalize a base URL (strip whitespace/trailing slashes)."""
    cleaned = url.strip().rstrip("/")
    return cleaned


def is_valid_base_url(url: str) -> bool:
    """Return True when the URL looks like an http(s) base URL."""
    try:
        parsed = urlparse(url.strip())
    except (ValueError, AttributeError):
        return False
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def config_dir() -> Path:
    """Return the per-user llmterm config directory."""
    return Path.home() / ".llmterm"


def config_path() -> Path:
    """Return the path of the URL-only config file (never stores keys)."""
    return config_dir() / "config.json"


def load_custom_specs() -> list[EndpointSpec]:
    """Load user-saved custom endpoint URLs (secrets are never stored)."""
    path = config_path()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    items = raw.get("custom_endpoints", []) if isinstance(raw, dict) else []
    specs: list[EndpointSpec] = []
    if not isinstance(items, list):
        return []
    for item in items:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        base_url = item.get("base_url")
        key_env = item.get("key_env")
        # Deliberately ignore any "api_key"/"key" payload, even if present
        # in an edited file: secrets must never be persisted.
        if not isinstance(name, str) or not isinstance(base_url, str):
            continue
        if not name.strip() or not is_valid_base_url(base_url):
            continue
        if key_env is not None and (not isinstance(key_env, str) or not key_env.strip()):
            continue
        specs.append(
            EndpointSpec(
                key=f"custom:{name.strip().lower().replace(' ', '-')}",
                name=name.strip(),
                base_url=normalize_base_url(base_url),
                key_env=key_env.strip() if isinstance(key_env, str) and key_env.strip() else None,
                requires_auth=True,
                help="Saved custom endpoint. API key is always re-entered, never saved.",
                is_custom=True,
            )
        )
    return specs


def save_custom_spec(name: str, base_url: str, key_env: str | None = None) -> EndpointSpec:
    """Persist a custom endpoint URL (never a key). Returns the new spec."""
    cleaned_name = name.strip()
    cleaned_url = normalize_base_url(base_url)
    if not cleaned_name:
        raise ValueError("Custom endpoint name must not be empty.")
    if not is_valid_base_url(cleaned_url):
        raise ValueError(f"Invalid base URL: {base_url!r}. Use http(s)://host[:port][/v1].")
    cleaned_env = key_env.strip() if isinstance(key_env, str) and key_env.strip() else None

    directory = config_dir()
    directory.mkdir(parents=True, exist_ok=True)
    path = config_path()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raw = {}
    except (OSError, ValueError):
        raw = {}
    items = raw.get("custom_endpoints")
    if not isinstance(items, list):
        items = []
    # Replace same-name entries; keep file small and auditable.
    items = [i for i in items if not (isinstance(i, dict) and i.get("name") == cleaned_name)]
    items.append({"name": cleaned_name, "base_url": cleaned_url, "key_env": cleaned_env})
    # Never write keys: strip any accidental secret fields defensively.
    for entry in items:
        if isinstance(entry, dict):
            entry.pop("api_key", None)
            entry.pop("apiKey", None)
            entry.pop("key", None)
    raw["custom_endpoints"] = items
    path.write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")

    return EndpointSpec(
        key=f"custom:{cleaned_name.lower().replace(' ', '-')}",
        name=cleaned_name,
        base_url=cleaned_url,
        key_env=cleaned_env,
        requires_auth=True,
        help="Saved custom endpoint. API key is always re-entered, never saved.",
        is_custom=True,
    )


def list_all_specs() -> list[EndpointSpec]:
    """Return built-in specs plus user-saved custom URL specs."""
    return [*BUILTIN_SPECS, *load_custom_specs()]


def find_spec(key_or_name: str) -> EndpointSpec | None:
    """Find a spec by key or case-insensitive name."""
    needle = key_or_name.strip().lower()
    for spec in list_all_specs():
        if spec.key.lower() == needle or spec.name.lower() == needle:
            return spec
    return None


def env_overrides() -> dict[str, str]:
    """Read supported LLMTERM_* environment overrides (URLs only, no keys)."""
    overrides: dict[str, str] = {}
    base_url = os.environ.get("LLMTERM_BASE_URL", "").strip()
    if base_url:
        overrides["base_url"] = base_url
    return overrides


def env_template() -> str:
    """Render a `.env` template for llmterm (variable names only, no values).

    Derived from the endpoint registry so new providers are picked up
    automatically. Commented out so nothing is active until the user edits it.
    """
    lines = [
        "# llmterm .env template.",
        "# API keys are read from the environment and are NEVER stored by llmterm:",
        "# not in config files, not in logs, not on disk.",
        "#",
        "# If you installed llmterm with `uv tool install`, save this output as",
        "# `.env` in the directory where you run `llmterm`, or set these",
        "# variables in your shell instead.",
        "",
    ]
    seen: set[str] = set()
    for spec in BUILTIN_SPECS:
        if not spec.key_env or spec.key_env in seen:
            continue
        seen.add(spec.key_env)
        scope = "required" if spec.requires_auth else "optional; only when the server requires auth"
        lines.append(f"# {spec.name} ({scope})")
        lines.append(f"#{spec.key_env}=your-key")
        lines.append("")
    lines += [
        "# Optional override for any endpoint base URL (URLs only, never keys).",
        "#LLMTERM_BASE_URL=https://your-deployment.example.com/v1",
        "",
    ]
    return "\n".join(lines)
