import logging
import os
from typing import Dict, Optional

import yaml
from dotenv import load_dotenv

# Local development settings live in .env. Existing process variables keep
# precedence so deployments can still inject configuration normally.
load_dotenv(override=False)


def load_config(config_path: str = "config.yaml") -> Dict:
    """Loads the configuration from the specified YAML file."""
    try:
        with open(config_path, "r") as f:
            config_data = yaml.safe_load(f)
            if not isinstance(config_data, dict):
                print(f"Error: Configuration file {config_path} did not load as a dictionary.")
                exit(1)
            # Convert logging level string to actual level
            log_level_str = config_data.get("logging_level", "INFO").upper()
            config_data["logging_level"] = getattr(logging, log_level_str, logging.INFO)
        return config_data
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {config_path}")
        exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing YAML in {config_path}: {e}")
        exit(1)
    except AttributeError:
        print(f"Error: Invalid logging level '{log_level_str}' in config file")
        exit(1)
    except KeyError as e:
        print(f"Error: Missing key in config file: {e}")
        exit(1)
    except Exception as e:
        print(f"An unexpected error occurred while loading config: {e}")
        exit(1)


# Load configuration at the start when this module is imported
config = load_config()


def get_llm_provider() -> str:
    """Return the normalized LLM backend selected by environment or config."""
    raw_provider = os.getenv("LLM_PROVIDER", str(config.get("llm_provider", "openrouter")))
    provider = raw_provider.strip().lower().replace("-", "_")
    if provider == "lmstudio":
        return "lm_studio"
    return provider


def get_llm_base_url(provider: Optional[str] = None) -> str:
    """Return the OpenAI-compatible API base URL for the active backend."""
    provider = provider or get_llm_provider()
    if provider == "lm_studio":
        configured_url = config.get("lmstudio_base_url") or config.get("lm_studio_base_url", "http://localhost:1234/v1")
        return os.getenv(
            "LMSTUDIO_BASE_URL",
            os.getenv("LM_STUDIO_BASE_URL", str(configured_url)),
        ).rstrip("/")
    return os.getenv(
        "OPENROUTER_BASE_URL", str(config.get("openrouter_base_url", "https://openrouter.ai/api/v1"))
    ).rstrip("/")


def get_llm_api_key(provider: Optional[str] = None) -> Optional[str]:
    """Read provider credentials from environment variables only."""
    provider = provider or get_llm_provider()
    if provider == "lm_studio":
        return os.getenv("LMSTUDIO_API_KEY") or os.getenv("LM_STUDIO_API_KEY")
    return os.getenv("OPENROUTER_API_KEY")


def get_configured_llm_model(provider: Optional[str] = None) -> str:
    """Return the configured model without mixing provider-specific defaults."""
    provider = provider or get_llm_provider()
    generic_override = os.getenv("LLM_MODEL")
    if generic_override:
        return generic_override.strip()
    if provider == "lm_studio":
        return (
            os.getenv("LMSTUDIO_MODEL") or os.getenv("LM_STUDIO_MODEL") or config.get("lm_studio_model", "")
        ).strip()
    return os.getenv("OPENROUTER_MODEL", str(config.get("llm_model", ""))).strip()


def get_llm_request_timeout_seconds(provider: Optional[str] = None) -> float:
    """Return a provider-specific request timeout with a safe numeric fallback."""
    provider = provider or get_llm_provider()
    if provider == "lm_studio":
        default = config.get("lmstudio_request_timeout_seconds", config.get("lm_studio_request_timeout_seconds", 300))
        raw_timeout = os.getenv("LMSTUDIO_REQUEST_TIMEOUT_SECONDS", str(default))
    else:
        default = config.get("llm_request_timeout_seconds", 30)
        raw_timeout = os.getenv("OPENROUTER_REQUEST_TIMEOUT_SECONDS", str(default))
    try:
        return float(raw_timeout)
    except (TypeError, ValueError):
        return float(default)


def is_model_fallback_enabled(provider: Optional[str] = None) -> bool:
    """Keep LM Studio pinned to one model unless local fallback is opted in."""
    provider = provider or get_llm_provider()
    if provider != "lm_studio":
        return True
    configured = str(config.get("lmstudio_enable_fallback", config.get("lm_studio_enable_fallback", False)))
    value = os.getenv("LMSTUDIO_ENABLE_FALLBACK", configured)
    return value.strip().lower() in {"1", "true", "yes", "on"}


# Example of accessing config values (optional, for clarity)
# print(f"LLM Model from config: {config.get('llm_model')}")
