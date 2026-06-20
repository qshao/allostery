from __future__ import annotations

from allostery.cli_errors import BACKEND_ERROR, USER_ERROR, exit_code_for
from allostery.config import ConfigError


def test_value_and_config_and_missing_file_are_user_errors() -> None:
    assert exit_code_for(ValueError("bad")) == USER_ERROR
    assert exit_code_for(ConfigError("bad config")) == USER_ERROR
    assert exit_code_for(FileNotFoundError("nope")) == USER_ERROR


def test_import_and_network_errors_are_backend_errors() -> None:
    assert exit_code_for(ImportError("no anthropic")) == BACKEND_ERROR
    assert exit_code_for(ConnectionError("refused")) == BACKEND_ERROR
    assert exit_code_for(OSError("socket")) == BACKEND_ERROR


def test_unexpected_returns_none() -> None:
    assert exit_code_for(KeyError("x")) is None


def test_chases_cause_one_level() -> None:
    inner = ImportError("no openai")
    outer = RuntimeError("workflow failed")
    outer.__cause__ = inner
    assert exit_code_for(outer) == BACKEND_ERROR
