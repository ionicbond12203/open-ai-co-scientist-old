"""Keep the default offline suite independent from a developer's .env file."""

import os

# Individual LM Studio tests opt in with monkeypatch. All other tests retain the
# historical OpenRouter behavior even when local development uses LM Studio.
os.environ["LLM_PROVIDER"] = "openrouter"
