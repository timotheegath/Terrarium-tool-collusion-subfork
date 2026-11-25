from __future__ import annotations

import os
from typing import Any, Optional

from dotenv import load_dotenv

from terrarium.llm.clients.openai_client import OpenAIClient


_DEFAULT_API_KEY_ENV_VAR = "OPENAI_COMPATIBLE_API_KEY"
_DEFAULT_BASE_URL_ENV_VAR = "OPENAI_COMPATIBLE_BASE_URL"
_DEFAULT_ENV_PREFIX = "OPENAI_COMPATIBLE"


class OpenAICompatibleClient(OpenAIClient):
    """
    Client for any OpenAI-compatible endpoint (custom base URL + API key).

    Talks to the provider through the OpenAI SDK's Responses API, exactly like
    ``OpenAIClient``, but points at a user-supplied base URL. Set both in the
    ``.env`` file:

    - ``OPENAI_COMPATIBLE_BASE_URL`` (e.g. ``https://my-gateway.example.com/v1``)
    - ``OPENAI_COMPATIBLE_API_KEY``

    Both can be overridden per instance or via the ``openai_compatible`` block
    of the LLM config (``base_url`` / ``base_url_env_var`` / ``api_key`` /
    ``api_key_env_var``).

    Unlike the other ``OpenAIClient`` subclasses, this provider streams (SSE)
    and strips ``reasoning`` items from the request input by default; both can
    be turned off per instance or via the ``openai_compatible`` config block
    (``stream`` / ``strip_reasoning_content``). Endpoints that reject streaming
    (400/404/405/422/501) fall back to a blocking call, and the read timeout
    (``request_timeout``, default 180s) then applies per-read between SSE
    chunks instead of capping total generation time.

    Retry/backoff settings resolve in order: constructor kwargs (fed from the
    ``openai_compatible`` config block) -> ``OPENAI_COMPATIBLE_*`` environment
    variables -> the ``OpenAIClient`` defaults:

    - ``OPENAI_COMPATIBLE_MAX_RETRIES`` (default: 6)
    - ``OPENAI_COMPATIBLE_RETRY_INITIAL_S`` (default: 0.5)
    - ``OPENAI_COMPATIBLE_RETRY_MAX_S`` (default: 20)
    - ``OPENAI_COMPATIBLE_RETRY_MULTIPLIER`` (default: 2)
    - ``OPENAI_COMPATIBLE_RETRY_JITTER`` (default: 0.1)
    """

    def __init__(
        self,
        *,
        api_key: Optional[Any] = None,
        api_key_env_var: str = _DEFAULT_API_KEY_ENV_VAR,
        base_url: Optional[str] = None,
        base_url_env_var: str = _DEFAULT_BASE_URL_ENV_VAR,
        timeout: float = 180.0,
        base_model: Optional[str] = None,
        provider_name: str = "OpenAI-compatible",
        stream: bool = True,
        strip_reasoning_content: bool = True,
        max_retries: Optional[int] = None,
        retry_initial_s: Optional[float] = None,
        retry_max_s: Optional[float] = None,
        retry_multiplier: Optional[float] = None,
        retry_jitter: Optional[float] = None,
    ):
        load_dotenv(override=True)
        resolved_api_key = api_key or os.getenv(api_key_env_var)
        if not resolved_api_key:
            raise ValueError(
                f"{provider_name} API key not found. Set {api_key_env_var} in .env file"
            )

        resolved_base_url = base_url or os.getenv(base_url_env_var)
        if not resolved_base_url:
            raise ValueError(
                f"{provider_name} base URL not found. Set {base_url_env_var} in .env file"
            )

        super().__init__(
            api_key=resolved_api_key,
            api_key_env_var=api_key_env_var,
            base_url=resolved_base_url,
            timeout=timeout,
            provider_name=provider_name,
            capability_model=base_model,
            stream=stream,
            strip_reasoning_content=strip_reasoning_content,
            max_retries=max_retries,
            retry_initial_s=retry_initial_s,
            retry_max_s=retry_max_s,
            retry_multiplier=retry_multiplier,
            retry_jitter=retry_jitter,
            env_prefix=_DEFAULT_ENV_PREFIX,
        )
