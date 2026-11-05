"""Tests for main helpers: selection parsing, endpoint building, CLI."""

import pytest

from llmterm import __version__, endpoints, main
from llmterm.endpoints import EndpointSpec


def test_parse_selection_index():
    assert main.parse_selection_index("1", 3) == 0
    assert main.parse_selection_index(" 3 ", 3) == 2
    assert main.parse_selection_index("0", 3) is None  # no -1 wrap
    assert main.parse_selection_index("4", 3) is None
    assert main.parse_selection_index("abc", 3) is None
    assert main.parse_selection_index("", 3) is None
    assert main.parse_selection_index("-1", 3) is None


def test_build_endpoint_rejects_bad_url():
    spec = EndpointSpec(key="t", name="T", base_url="http://x/v1")
    assert main.build_endpoint(spec, base_url_override="not-a-url") is None


def test_build_endpoint_missing_required_key_no_prompt(monkeypatch):
    monkeypatch.delenv("TEST_MISSING_KEY_XYZ", raising=False)
    spec = EndpointSpec(
        key="t",
        name="T",
        base_url="https://example.com/v1",
        key_env="TEST_MISSING_KEY_XYZ",
        requires_auth=True,
    )
    assert main.build_endpoint(spec, prompt_for_key=False) is None


def test_build_endpoint_uses_env_key(monkeypatch):
    monkeypatch.setenv("TEST_PRESENT_KEY_XYZ", "secret")
    spec = EndpointSpec(
        key="t",
        name="T",
        base_url="https://example.com/v1",
        key_env="TEST_PRESENT_KEY_XYZ",
        requires_auth=True,
    )
    endpoint = main.build_endpoint(spec, prompt_for_key=False)
    assert endpoint is not None
    assert endpoint.api_key == "secret"


def test_build_endpoint_local_needs_no_key(monkeypatch):
    monkeypatch.delenv("UNSLOTH_STUDIO_API_KEY", raising=False)
    spec = endpoints.find_spec("ollama")
    assert spec is not None
    endpoint = main.build_endpoint(spec, prompt_for_key=False)
    assert endpoint is not None
    assert endpoint.api_key is None


def test_parser_has_no_api_key_flag():
    parser = main.create_parser()
    actions = {a.dest for a in parser._actions}
    assert "api_key" not in actions  # keys must never ride the CLI
    assert "api_key_env" in actions
    assert "endpoint" in actions
    assert "list_endpoints" in actions


def test_parser_parses_non_interactive():
    args = main.create_parser().parse_args(["--endpoint", "ollama", "--model", "m", "--no-save"])
    assert args.endpoint == "ollama"
    assert args.model == "m"
    assert args.no_save is True


def test_version_matches_package():
    assert __version__ == "0.2.0"


def test_show_env_template_flag_prints_and_exits(capsys):
    assert "show_env_template" in {a.dest for a in main.create_parser()._actions}
    with pytest.raises(SystemExit) as exc_info:
        main.main(["--show-env-template"])
    assert exc_info.value.code == 0
    out = capsys.readouterr().out
    assert "OPENAI_API_KEY" in out
    assert "NEVER stored" in out
