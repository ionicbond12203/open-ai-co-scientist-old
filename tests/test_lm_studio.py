"""Offline coverage for the LM Studio OpenAI-compatible backend."""

from unittest.mock import MagicMock, patch

import app.utils as utils
from app.config import (
    get_configured_llm_model,
    get_llm_base_url,
    get_llm_provider,
    get_llm_request_timeout_seconds,
    is_model_fallback_enabled,
)
from app.models import ResearchGoal


def _completion(content: str):
    completion = MagicMock()
    choice = MagicMock()
    choice.message.content = content
    completion.choices = [choice]
    return completion


def test_lm_studio_environment_names_are_supported(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "lm-studio")
    monkeypatch.setenv("LMSTUDIO_BASE_URL", "http://lm-studio.test:1234/v1/")
    monkeypatch.setenv("LMSTUDIO_MODEL", "local/chat-model")
    monkeypatch.setenv("LMSTUDIO_REQUEST_TIMEOUT_SECONDS", "300")
    monkeypatch.setenv("LMSTUDIO_ENABLE_FALLBACK", "false")

    assert get_llm_provider() == "lm_studio"
    assert get_llm_base_url() == "http://lm-studio.test:1234/v1"
    assert get_configured_llm_model() == "local/chat-model"
    assert get_llm_request_timeout_seconds() == 300
    assert is_model_fallback_enabled() is False
    assert ResearchGoal("test").llm_model == "local/chat-model"


def test_fetch_lm_studio_models_filters_non_chat_models_and_prefers_small_models(monkeypatch):
    monkeypatch.setenv("LMSTUDIO_BASE_URL", "http://lm-studio.test:1234/v1")
    monkeypatch.setenv("LMSTUDIO_API_KEY", "local-test-key")
    response = MagicMock()
    response.json.return_value = {
        "data": [
            {"id": "qwen/qwen3.8-27b"},
            {"id": "text-embedding-qwen3-embedding-8b"},
            {"id": "qwen/qwen3.5-9b"},
            {"id": "qwen/qwen3.5-9b"},
            {"id": "qwen3-reranker-4b"},
            {"id": "qwen/qwen3-coder-next"},
        ]
    }

    with patch.object(utils.requests, "get", return_value=response) as mock_get:
        models = utils.fetch_lm_studio_models()

    assert models == ["qwen/qwen3.5-9b", "qwen/qwen3.8-27b", "qwen/qwen3-coder-next"]
    mock_get.assert_called_once_with(
        "http://lm-studio.test:1234/v1/models",
        headers={"User-Agent": "open-ai-co-scientist/1.0", "Authorization": "Bearer local-test-key"},
        timeout=10,
    )


def test_call_llm_uses_lm_studio_without_requiring_api_key(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "lm_studio")
    monkeypatch.setenv("LMSTUDIO_BASE_URL", "http://lm-studio.test:1234/v1")
    monkeypatch.setenv("LMSTUDIO_MODEL", "local/chat-model")
    monkeypatch.setenv("LMSTUDIO_REQUEST_TIMEOUT_SECONDS", "300")
    monkeypatch.setenv("LMSTUDIO_CONNECT_TIMEOUT_SECONDS", "10")
    monkeypatch.delenv("LMSTUDIO_API_KEY", raising=False)
    monkeypatch.delenv("LM_STUDIO_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    with patch.object(utils, "OpenAI") as mock_openai:
        mock_openai.return_value.chat.completions.create.return_value = _completion("LOCAL RESPONSE")
        result = utils.call_llm("prompt", temperature=0.4)

    assert result == "LOCAL RESPONSE"
    assert mock_openai.call_args.kwargs["base_url"] == "http://lm-studio.test:1234/v1"
    assert mock_openai.call_args.kwargs["api_key"]
    timeout = mock_openai.call_args.kwargs["timeout"]
    assert timeout.connect == 10
    assert timeout.read == 300
    mock_openai.return_value.chat.completions.create.assert_called_once_with(
        model="local/chat-model",
        messages=[{"role": "user", "content": "prompt"}],
        temperature=0.4,
    )


def test_lm_studio_fallbacks_stay_on_lm_studio(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "lm_studio")
    monkeypatch.setenv("LMSTUDIO_MODEL", "missing/model")
    monkeypatch.setenv("LMSTUDIO_ENABLE_FALLBACK", "true")

    def create(model=None, messages=None, temperature=None):
        if model == "working/model":
            return _completion("RECOVERED LOCALLY")
        raise Exception("not a valid model")

    client = MagicMock()
    client.chat.completions.create.side_effect = create
    with (
        patch.object(utils, "OpenAI", return_value=client),
        patch.object(utils, "fetch_lm_studio_models", return_value=["missing/model", "working/model"]) as mock_fetch,
        patch.object(utils, "fetch_free_models") as mock_openrouter_fetch,
    ):
        result = utils.call_llm("prompt")

    assert result == "RECOVERED LOCALLY"
    mock_fetch.assert_called_once()
    mock_openrouter_fetch.assert_not_called()


def test_lm_studio_pinned_model_does_not_fallback_on_timeout(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "lm_studio")
    monkeypatch.setenv("LMSTUDIO_MODEL", "qwen/qwen3.8-27b")
    monkeypatch.setenv("LMSTUDIO_ENABLE_FALLBACK", "false")
    client = MagicMock()
    client.chat.completions.create.side_effect = TimeoutError("Request timed out")

    with (
        patch.object(utils, "OpenAI", return_value=client),
        patch.object(utils, "fetch_lm_studio_models") as mock_fetch,
    ):
        result = utils.call_llm("prompt")

    assert result.startswith("Error: Model provider timed out")
    assert client.chat.completions.create.call_count == 1
    mock_fetch.assert_not_called()


def test_lm_studio_api_key_is_redacted(monkeypatch):
    key = "lm-studio-secret-for-test"
    monkeypatch.setenv("LMSTUDIO_API_KEY", key)

    assert key not in utils.redact_secrets(f"provider echoed {key}")
