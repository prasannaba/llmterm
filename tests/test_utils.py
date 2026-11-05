"""Tests for utils: prompts, cleanup, filenames, markdown."""

import pytest

from llmterm import utils


def test_clean_response_rewrites_escapes_outside_code():
    out = utils.clean_response(r"Use \[x\] and \(y\) here")
    assert "$$x$$" in out and "$y$" in out


def test_clean_response_leaves_code_fences_alone():
    src = "```\n" + r"\[keep\] \(keep\)" + "\n```"
    assert utils.clean_response(src) == src


def test_clean_response_idempotent():
    src = r"a \[b\] c \(d\)"
    assert utils.clean_response(utils.clean_response(src)) == utils.clean_response(src)


def test_set_system_prompt_eof_returns_default(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda *a, **k: (_ for _ in ()).throw(EOFError))
    assert utils.set_system_prompt() == ""


def test_set_system_prompt_yes_empty_keeps_default(monkeypatch):
    answers = iter(["y", "   "])
    monkeypatch.setattr("builtins.input", lambda *a, **k: next(answers))
    assert utils.set_system_prompt() == ""


def test_sanitize_model_filename():
    sanitized = utils.sanitize_model_for_filename("../../evil")
    assert "/" not in sanitized and ".." not in sanitized
    assert sanitized.endswith("evil")
    assert utils.sanitize_model_for_filename("con") == "con-model"
    assert utils.sanitize_model_for_filename("   ") == "unknown-model"
    assert utils.sanitize_model_for_filename("a:b/c\\d") == "a-b-c-d"
    long_name = "m" * 200
    assert len(utils.sanitize_model_for_filename(long_name)) <= 100


def test_save_response_writes_file_and_refuses_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(utils.Path, "home", classmethod(lambda cls: tmp_path))
    path = utils.save_response_to_markdown("my-model", "hello", "world")
    assert path.exists()
    text = path.read_text(encoding="utf-8")
    assert "hello" in text and "world" in text
    with pytest.raises(ValueError):
        utils.save_response_to_markdown("my-model", "hello", "   ")


def test_save_response_never_writes_keys(tmp_path, monkeypatch):
    monkeypatch.setattr(utils.Path, "home", classmethod(lambda cls: tmp_path))
    path = utils.save_response_to_markdown("m", "prompt", "response body")
    blob = path.read_text(encoding="utf-8")
    assert "sk-" not in blob


def test_get_api_key_from_env(monkeypatch):
    monkeypatch.setenv("TEST_LLM_KEY", "  secret-value  ")
    assert (
        utils.get_api_key(env_var="TEST_LLM_KEY", endpoint_name="X", required=True, prompt=False)
        == "secret-value"
    )


def test_get_api_key_missing_required_no_prompt(monkeypatch):
    monkeypatch.delenv("TEST_LLM_KEY_MISSING", raising=False)
    assert (
        utils.get_api_key(
            env_var="TEST_LLM_KEY_MISSING",
            endpoint_name="X",
            required=True,
            prompt=False,
        )
        is None
    )


def test_get_api_key_optional_returns_none(monkeypatch):
    monkeypatch.delenv("TEST_LLM_KEY_OPT", raising=False)
    assert (
        utils.get_api_key(
            env_var="TEST_LLM_KEY_OPT",
            endpoint_name="X",
            required=False,
            prompt=False,
        )
        is None
    )


def test_get_unsloth_key_defaults_none(monkeypatch):
    monkeypatch.delenv("UNSLOTH_STUDIO_API_KEY", raising=False)
    assert utils.get_unsloth_api_key() is None
