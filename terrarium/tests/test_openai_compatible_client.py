import importlib
import os
import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch


class _FakeEventStream:
    """Iterable SSE event stream; raises ``error`` once ``events`` are consumed."""

    def __init__(self, events=(), error=None):
        self._events = list(events)
        self._error = error

    def __iter__(self):
        return self

    def __next__(self):
        if self._events:
            return self._events.pop(0)
        if self._error is not None:
            raise self._error
        raise StopIteration


class _FakeResponsesAPI:
    def __init__(self):
        self.calls = []
        self.next_response = SimpleNamespace(output=[], output_text="", usage=None)
        self.queued = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        item = self.queued.pop(0) if self.queued else self.next_response
        if isinstance(item, Exception):
            raise item
        if kwargs.get("stream"):
            if isinstance(item, list):
                return iter(item)
            if isinstance(item, _FakeEventStream):
                return item
            return iter([SimpleNamespace(type="response.completed", response=item)])
        return item


class _FakeOpenAI:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.responses = _FakeResponsesAPI()
        _FakeOpenAI.instances.append(self)


class _FakeMessage(dict):
    pass


class _FakeFunctionCallOutput:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def _fake_sys_modules():
    fake_openai = types.ModuleType("openai")
    fake_openai.OpenAI = _FakeOpenAI

    class _APIConnectionError(Exception):
        pass

    class _APIStatusError(Exception):
        pass

    class _APITimeoutError(Exception):
        pass

    class _RateLimitError(Exception):
        pass

    fake_openai.APIConnectionError = _APIConnectionError
    fake_openai.APIStatusError = _APIStatusError
    fake_openai.APITimeoutError = _APITimeoutError
    fake_openai.RateLimitError = _RateLimitError

    fake_oai_types = types.ModuleType("openai.types")
    fake_oai_types_responses = types.ModuleType("openai.types.responses")
    fake_oai_types_response_input = types.ModuleType(
        "openai.types.responses.response_input_item_param"
    )
    fake_oai_types_response_input.Message = _FakeMessage
    fake_oai_types_response_input.FunctionCallOutput = _FakeFunctionCallOutput

    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda override=True: None

    return {
        "openai": fake_openai,
        "openai.types": fake_oai_types,
        "openai.types.responses": fake_oai_types_responses,
        "openai.types.responses.response_input_item_param": fake_oai_types_response_input,
        "dotenv": fake_dotenv,
    }


def _pop_client_modules():
    for module_name in (
        "terrarium.llm.clients.openai_client",
        "terrarium.llm.clients.openai_compatible_client",
    ):
        sys.modules.pop(module_name, None)


def _import_openai_compatible_client_with_mocks():
    _pop_client_modules()
    with patch.dict(sys.modules, _fake_sys_modules()):
        module = importlib.import_module(
            "terrarium.llm.clients.openai_compatible_client"
        )
    return module


class OpenAICompatibleClientTests(unittest.TestCase):
    def setUp(self):
        _FakeOpenAI.instances = []

    def test_uses_env_base_url_and_api_key(self):
        module = _import_openai_compatible_client_with_mocks()
        OpenAICompatibleClient = module.OpenAICompatibleClient

        with patch.dict(
            os.environ,
            {
                "OPENAI_COMPATIBLE_API_KEY": "compat-key",
                "OPENAI_COMPATIBLE_BASE_URL": "https://my-gateway.example.com/v1",
            },
            clear=True,
        ):
            OpenAICompatibleClient()

        kwargs = _FakeOpenAI.instances[-1].kwargs
        self.assertEqual(kwargs["api_key"], "compat-key")
        self.assertEqual(kwargs["base_url"], "https://my-gateway.example.com/v1/")
        self.assertEqual(kwargs["timeout"], 180.0)

    def test_custom_env_var_names_are_used(self):
        module = _import_openai_compatible_client_with_mocks()
        OpenAICompatibleClient = module.OpenAICompatibleClient

        with patch.dict(
            os.environ,
            {
                "MY_PROVIDER_KEY": "custom-key",
                "MY_PROVIDER_URL": "https://custom.example.com/v1",
            },
            clear=True,
        ):
            OpenAICompatibleClient(
                api_key_env_var="MY_PROVIDER_KEY",
                base_url_env_var="MY_PROVIDER_URL",
            )

        kwargs = _FakeOpenAI.instances[-1].kwargs
        self.assertEqual(kwargs["api_key"], "custom-key")
        self.assertEqual(kwargs["base_url"], "https://custom.example.com/v1/")

    def test_explicit_args_override_env(self):
        module = _import_openai_compatible_client_with_mocks()
        OpenAICompatibleClient = module.OpenAICompatibleClient

        with patch.dict(
            os.environ,
            {
                "OPENAI_COMPATIBLE_API_KEY": "env-key",
                "OPENAI_COMPATIBLE_BASE_URL": "https://env.example.com/v1",
            },
            clear=True,
        ):
            OpenAICompatibleClient(
                api_key="explicit-key",
                base_url="https://explicit.example.com/v1",
            )

        kwargs = _FakeOpenAI.instances[-1].kwargs
        self.assertEqual(kwargs["api_key"], "explicit-key")
        self.assertEqual(kwargs["base_url"], "https://explicit.example.com/v1/")

    def test_missing_api_key_raises(self):
        module = _import_openai_compatible_client_with_mocks()
        OpenAICompatibleClient = module.OpenAICompatibleClient

        with patch.dict(
            os.environ,
            {"OPENAI_COMPATIBLE_BASE_URL": "https://my-gateway.example.com/v1"},
            clear=True,
        ):
            with self.assertRaises(ValueError) as ctx:
                OpenAICompatibleClient()

        self.assertIn("OPENAI_COMPATIBLE_API_KEY", str(ctx.exception))

    def test_missing_base_url_raises(self):
        module = _import_openai_compatible_client_with_mocks()
        OpenAICompatibleClient = module.OpenAICompatibleClient

        with patch.dict(
            os.environ,
            {"OPENAI_COMPATIBLE_API_KEY": "compat-key"},
            clear=True,
        ):
            with self.assertRaises(ValueError) as ctx:
                OpenAICompatibleClient()

        self.assertIn("OPENAI_COMPATIBLE_BASE_URL", str(ctx.exception))

    def test_non_reasoning_model_keeps_temperature(self):
        module = _import_openai_compatible_client_with_mocks()
        OpenAICompatibleClient = module.OpenAICompatibleClient

        with patch.dict(
            os.environ,
            {
                "OPENAI_COMPATIBLE_API_KEY": "compat-key",
                "OPENAI_COMPATIBLE_BASE_URL": "https://my-gateway.example.com/v1",
            },
            clear=True,
        ):
            client = OpenAICompatibleClient()
            sdk_instance = _FakeOpenAI.instances[-1]
            context = client.init_context("system", "user")
            client.generate_response(
                input=context,
                params={
                    "model": "my-local-chat-model",
                    "max_output_tokens": 128,
                    "temperature": 0.7,
                },
            )

        call = sdk_instance.responses.calls[-1]
        self.assertEqual(call["model"], "my-local-chat-model")
        self.assertEqual(call["temperature"], 0.7)

    def _make_client(self, module, **kwargs):
        with patch.dict(
            os.environ,
            {
                "OPENAI_COMPATIBLE_API_KEY": "compat-key",
                "OPENAI_COMPATIBLE_BASE_URL": "https://my-gateway.example.com/v1",
            },
            clear=True,
        ):
            return module.OpenAICompatibleClient(**kwargs)

    def _status_error(self, status_code):
        exc = Exception(f"HTTP {status_code}")
        exc.status_code = status_code
        return exc

    def test_env_prefix_uses_openai_compatible_env_vars(self):
        module = _import_openai_compatible_client_with_mocks()
        OpenAICompatibleClient = module.OpenAICompatibleClient

        with patch.dict(
            os.environ,
            {
                "OPENAI_COMPATIBLE_API_KEY": "compat-key",
                "OPENAI_COMPATIBLE_BASE_URL": "https://my-gateway.example.com/v1",
                "OPENAI_COMPATIBLE_MAX_RETRIES": "9",
                "OPENAI_COMPATIBLE_RETRY_INITIAL_S": "1.5",
                "OPENAI_COMPATIBLE_RETRY_MAX_S": "30",
                "OPENAI_COMPATIBLE_RETRY_MULTIPLIER": "3",
                "OPENAI_COMPATIBLE_RETRY_JITTER": "0.25",
                # Generic OpenAI_* retry vars must be ignored by this provider.
                "OPENAI_MAX_RETRIES": "2",
                "OPENAI_RETRY_JITTER": "0.9",
            },
            clear=True,
        ):
            client = OpenAICompatibleClient()

        self.assertEqual(client._env_prefix, "OPENAI_COMPATIBLE")
        self.assertEqual(client._max_retries, 9)
        self.assertEqual(client._backoff_initial_s, 1.5)
        self.assertEqual(client._backoff_max_s, 30.0)
        self.assertEqual(client._backoff_multiplier, 3.0)
        self.assertEqual(client._backoff_jitter, 0.25)

    def test_retry_defaults_when_no_env_or_kwargs(self):
        module = _import_openai_compatible_client_with_mocks()
        client = self._make_client(module)

        self.assertEqual(client._max_retries, 6)
        self.assertEqual(client._backoff_initial_s, 0.5)
        self.assertEqual(client._backoff_max_s, 20.0)
        self.assertEqual(client._backoff_multiplier, 2.0)
        self.assertEqual(client._backoff_jitter, 0.1)

    def test_retry_constructor_args_beat_env_vars(self):
        module = _import_openai_compatible_client_with_mocks()
        OpenAICompatibleClient = module.OpenAICompatibleClient

        with patch.dict(
            os.environ,
            {
                "OPENAI_COMPATIBLE_API_KEY": "compat-key",
                "OPENAI_COMPATIBLE_BASE_URL": "https://my-gateway.example.com/v1",
                "OPENAI_COMPATIBLE_MAX_RETRIES": "9",
                "OPENAI_COMPATIBLE_RETRY_JITTER": "0.9",
            },
            clear=True,
        ):
            client = OpenAICompatibleClient(max_retries=3, retry_jitter=0.05, retry_max_s=99)

        self.assertEqual(client._max_retries, 3)
        self.assertEqual(client._backoff_jitter, 0.05)
        self.assertEqual(client._backoff_max_s, 99.0)

    def test_stream_and_strip_default_on(self):
        module = _import_openai_compatible_client_with_mocks()
        client = self._make_client(module)

        self.assertTrue(client.stream)
        self.assertTrue(client.strip_reasoning_content)

    def test_generate_response_streams_and_returns_completed_response(self):
        module = _import_openai_compatible_client_with_mocks()
        client = self._make_client(module)
        sdk_instance = _FakeOpenAI.instances[-1]

        fake_response = SimpleNamespace(
            output=[
                SimpleNamespace(
                    type="message",
                    content=[SimpleNamespace(type="output_text", text="streamed-text")],
                )
            ],
            output_text="streamed-text",
        )
        sdk_instance.responses.next_response = [
            SimpleNamespace(type="response.created", response=SimpleNamespace()),
            SimpleNamespace(type="response.completed", response=fake_response),
        ]

        context = client.init_context("system", "user")
        response, response_str = client.generate_response(
            input=context,
            params={"model": "my-local-chat-model", "max_output_tokens": 128},
        )

        self.assertIs(response, fake_response)
        self.assertEqual(response_str, "streamed-text")
        self.assertEqual(len(sdk_instance.responses.calls), 1)
        self.assertIs(sdk_instance.responses.calls[0]["stream"], True)

    def test_stream_reject_falls_back_to_blocking_call(self):
        module = _import_openai_compatible_client_with_mocks()
        client = self._make_client(module)
        sdk_instance = _FakeOpenAI.instances[-1]

        fake_response = SimpleNamespace(output=[], output_text="fallback-text")
        sdk_instance.responses.queued = [
            self._status_error(422),
            fake_response,
        ]

        context = client.init_context("system", "user")
        response, response_str = client.generate_response(
            input=context,
            params={"model": "my-local-chat-model"},
        )

        self.assertIs(response, fake_response)
        self.assertEqual(response_str, "fallback-text")
        calls = sdk_instance.responses.calls
        self.assertEqual(len(calls), 2)
        self.assertIs(calls[0]["stream"], True)
        self.assertNotIn("stream", calls[1])

    def test_stream_reject_ladder_ends_in_blocking_call_with_retries(self):
        module = _import_openai_compatible_client_with_mocks()
        client = self._make_client(module)
        sdk_instance = _FakeOpenAI.instances[-1]

        fake_response = SimpleNamespace(output=[], output_text="ladder-text")
        sdk_instance.responses.queued = [
            self._status_error(400),   # stream rejected
            self._status_error(429),   # single non-streaming retry fails (retryable)
            fake_response,             # plain blocking call succeeds
        ]

        context = client.init_context("system", "user")
        with patch("time.sleep"):
            response, response_str = client.generate_response(
                input=context,
                params={"model": "my-local-chat-model"},
            )

        self.assertIs(response, fake_response)
        self.assertEqual(response_str, "ladder-text")
        calls = sdk_instance.responses.calls
        self.assertEqual(len(calls), 3)
        self.assertIs(calls[0]["stream"], True)
        self.assertNotIn("stream", calls[1])
        self.assertNotIn("stream", calls[2])

    def test_generate_response_strips_reasoning_items_by_default(self):
        module = _import_openai_compatible_client_with_mocks()
        client = self._make_client(module)
        sdk_instance = _FakeOpenAI.instances[-1]
        sdk_instance.responses.next_response = SimpleNamespace(output=[], output_text="")

        client.generate_response(
            input=[
                {"type": "message", "role": "user", "content": "user"},
                SimpleNamespace(
                    type="reasoning",
                    id="rs_123",
                    summary=[],
                    status="completed",
                ),
                SimpleNamespace(
                    type="message",
                    role="assistant",
                    id="msg_123",
                    status="completed",
                    content=[],
                ),
            ],
            params={"model": "my-local-chat-model"},
        )

        call = sdk_instance.responses.calls[0]
        self.assertEqual(len(call["input"]), 2)
        self.assertNotIn("reasoning", [item["type"] for item in call["input"]])

    def test_generate_response_keeps_reasoning_items_when_strip_disabled(self):
        module = _import_openai_compatible_client_with_mocks()
        client = self._make_client(module, strip_reasoning_content=False)
        sdk_instance = _FakeOpenAI.instances[-1]
        sdk_instance.responses.next_response = SimpleNamespace(output=[], output_text="")

        client.generate_response(
            input=[
                {"type": "message", "role": "user", "content": "user"},
                SimpleNamespace(
                    type="reasoning",
                    id="rs_123",
                    summary=[],
                    status="completed",
                ),
            ],
            params={"model": "my-local-chat-model"},
        )

        call = sdk_instance.responses.calls[0]
        self.assertEqual(len(call["input"]), 2)
        self.assertEqual(call["input"][1]["type"], "reasoning")

    def test_utils_openai_compatible_block_overrides_env_and_defaults(self):
        import terrarium.utils as utils

        llm_config = {
            "provider": "openai-compatible",
            "openai_compatible": {
                "model": "my-local-model",
                "base_url": "https://cfg.example.com/v1",
                "api_key": "cfg-key",
                "request_timeout": 77,
                "stream": False,
                "strip_reasoning_content": False,
                "max_retries": 3,
                "retry_initial_s": 1.0,
                "retry_max_s": 9,
                "retry_multiplier": 4,
                # retry_jitter intentionally absent: env var must win over default.
            },
        }

        _pop_client_modules()
        with patch.dict(
            sys.modules,
            _fake_sys_modules(),
        ):
            with patch.dict(
                os.environ,
                {
                    "OPENAI_COMPATIBLE_MAX_RETRIES": "9",
                    "OPENAI_COMPATIBLE_RETRY_JITTER": "0.9",
                },
                clear=True,
            ):
                client = utils.get_client_instance(llm_config)
                client_cls = sys.modules[
                    "terrarium.llm.clients.openai_compatible_client"
                ].OpenAICompatibleClient

        self.assertIsInstance(client, client_cls)
        # Config block beats OPENAI_COMPATIBLE_* env vars.
        self.assertEqual(client._max_retries, 3)
        self.assertEqual(client._backoff_initial_s, 1.0)
        self.assertEqual(client._backoff_max_s, 9.0)
        self.assertEqual(client._backoff_multiplier, 4.0)
        # Env var beats the OpenAIClient default when the block omits the key.
        self.assertEqual(client._backoff_jitter, 0.9)
        # Transport keys are forwarded from the block.
        self.assertFalse(client.stream)
        self.assertFalse(client.strip_reasoning_content)
        kwargs = _FakeOpenAI.instances[-1].kwargs
        self.assertEqual(kwargs["api_key"], "cfg-key")
        self.assertEqual(kwargs["base_url"], "https://cfg.example.com/v1/")
        self.assertEqual(kwargs["timeout"], 77.0)

    def test_utils_openai_compatible_env_and_defaults_without_block_keys(self):
        import terrarium.utils as utils

        llm_config = {
            "provider": "openai_compat",
            "openai_compatible": {
                "model": "my-local-model",
            },
        }

        _pop_client_modules()
        with patch.dict(
            sys.modules,
            _fake_sys_modules(),
        ):
            with patch.dict(
                os.environ,
                {
                    "OPENAI_COMPATIBLE_API_KEY": "env-key",
                    "OPENAI_COMPATIBLE_BASE_URL": "https://env.example.com/v1",
                    "OPENAI_COMPATIBLE_MAX_RETRIES": "8",
                },
                clear=True,
            ):
                client = utils.get_client_instance(llm_config)
                client_cls = sys.modules[
                    "terrarium.llm.clients.openai_compatible_client"
                ].OpenAICompatibleClient

        self.assertIsInstance(client, client_cls)
        # Env var beats the default when the block omits the key.
        self.assertEqual(client._max_retries, 8)
        # Defaults apply for keys absent from both block and env.
        self.assertEqual(client._backoff_initial_s, 0.5)
        self.assertEqual(client._backoff_jitter, 0.1)
        # Block-less transport keys keep the provider defaults.
        self.assertTrue(client.stream)
        self.assertTrue(client.strip_reasoning_content)
        kwargs = _FakeOpenAI.instances[-1].kwargs
        self.assertEqual(kwargs["timeout"], 180.0)


if __name__ == "__main__":
    unittest.main()
