"""Tests for endpoint registry: validation, lookup, URL-only persistence."""

import json
from pathlib import Path

import pytest

from llmterm import endpoints
from llmterm.endpoints import env_template, is_valid_base_url, normalize_base_url


def test_normalize_strips_trailing_slash():
    assert normalize_base_url("https://example.com/v1///  ") == "https://example.com/v1"


def test_valid_urls():
    assert is_valid_base_url("http://localhost:8080/v1")
    assert is_valid_base_url("https://api.openai.com/v1")
    assert not is_valid_base_url("not-a-url")
    assert not is_valid_base_url("ftp://example.com/v1")
    assert not is_valid_base_url("")


def test_builtin_specs_have_valid_urls_or_none_with_help():
    for spec in endpoints.BUILTIN_SPECS:
        if spec.base_url is not None:
            assert is_valid_base_url(spec.base_url), spec
        if spec.requires_auth:
            assert spec.key_env, spec
            assert spec.help, spec


def test_find_spec_by_key_and_name():
    openai_spec = endpoints.find_spec("openai")
    assert openai_spec is not None and openai_spec.key == "openai"
    assert endpoints.find_spec("OpenAI") is not None
    ollama_spec = endpoints.find_spec("ollama")
    assert ollama_spec is not None and ollama_spec.name == "Ollama"
    assert endpoints.find_spec("no-such-endpoint") is None


def test_save_and_load_custom_url_only(tmp_path, monkeypatch):
    monkeypatch.setattr(endpoints.Path, "home", classmethod(lambda cls: tmp_path))
    spec = endpoints.save_custom_spec("My Cloud", "https://example.com/v1", "MY_KEY_ENV")
    assert spec.base_url == "https://example.com/v1"

    raw = json.loads((tmp_path / ".llmterm" / "config.json").read_text())
    blob = json.dumps(raw)
    assert "MY_KEY_ENV" in blob  # env var NAME may persist (not secret)
    assert "sk-" not in blob and "api_key" not in blob.lower()

    loaded = endpoints.load_custom_specs()
    assert any(s.name == "My Cloud" and s.base_url == "https://example.com/v1" for s in loaded)


def test_save_custom_never_persists_secret_fields(tmp_path, monkeypatch):
    monkeypatch.setattr(endpoints.Path, "home", classmethod(lambda cls: tmp_path))
    endpoints.save_custom_spec("Evil", "https://example.com/v1")
    path = tmp_path / ".llmterm" / "config.json"
    raw = json.loads(path.read_text())
    # Simulate a hand-edited file attempting to store a secret.
    raw["custom_endpoints"][0]["api_key"] = "sk-super-secret"
    path.write_text(json.dumps(raw))
    # Next save must strip the secret field.
    endpoints.save_custom_spec("Other", "https://other.example.com/v1")
    blob = (tmp_path / ".llmterm" / "config.json").read_text()
    assert "sk-super-secret" not in blob


def test_save_custom_rejects_bad_url():
    with pytest.raises(ValueError):
        endpoints.save_custom_spec("Bad", "not-a-url")
    with pytest.raises(ValueError):
        endpoints.save_custom_spec("  ", "https://example.com/v1")


def test_load_custom_ignores_malformed(tmp_path, monkeypatch):
    monkeypatch.setattr(endpoints.Path, "home", classmethod(lambda cls: tmp_path))
    cfg = tmp_path / ".llmterm"
    cfg.mkdir(parents=True, exist_ok=True)
    (cfg / "config.json").write_text(
        json.dumps({"custom_endpoints": ["junk", {"name": "", "base_url": "x"}]})
    )
    assert endpoints.load_custom_specs() == []


def test_env_template_covers_every_key_env():
    template = env_template()
    for spec in endpoints.BUILTIN_SPECS:
        if spec.key_env:
            assert spec.key_env in template
    assert "sk-" not in template


def test_env_example_lists_every_key_env():
    example = Path(__file__).resolve().parent.parent / ".env.example"
    content = example.read_text(encoding="utf-8")
    for spec in endpoints.BUILTIN_SPECS:
        if spec.key_env:
            assert spec.key_env in content, f"{spec.key_env} missing from .env.example"
