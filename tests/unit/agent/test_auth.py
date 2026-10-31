import pytest

from search_rl_env.agent.auth import load_api_key


def test_environment_key_takes_priority(monkeypatch, tmp_path):
    monkeypatch.setenv("MODEL_API_KEY", " env-key ")
    path = tmp_path / "key"
    path.write_text("file-key")
    assert load_api_key(key_file=path) == "env-key"


def test_configured_key_file(monkeypatch, tmp_path):
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    path = tmp_path / "key"
    path.write_text("file-key\n")
    monkeypatch.setenv("MODEL_API_KEY_FILE", str(path))
    assert load_api_key() == "file-key"


def test_missing_key_configuration(monkeypatch):
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    monkeypatch.delenv("MODEL_API_KEY_FILE", raising=False)
    with pytest.raises(ValueError, match="MODEL_API_KEY"):
        load_api_key()


def test_key_file_error_is_sanitized(monkeypatch, tmp_path):
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    with pytest.raises(ValueError) as error:
        load_api_key(key_file=tmp_path / "private-location")
    assert "private-location" not in str(error.value)
