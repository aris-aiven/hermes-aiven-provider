"""Aiven profile discovered from an isolated Hermes user-plugin directory."""

import io
import json
import os
import shutil
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import providers
import pytest
from providers import get_provider_profile


@pytest.mark.parametrize(
    "name", ["aiven", "aiven-ai-gateway", "aiven-gateway", "aiven_ai_gateway"]
)
def test_discovered_by_name_and_alias(name):
    from hermes_cli.providers import get_provider, normalize_provider

    p = get_provider_profile(name)
    assert p is not None
    assert p.name == "aiven"
    assert p.base_url == "https://ai.aiven.io/v1"
    assert normalize_provider(name) == "aiven"

    pdef = get_provider(name, allow_network=False)
    assert pdef is not None
    assert pdef.id == "aiven"
    assert pdef.base_url == "https://ai.aiven.io/v1"
    assert pdef.api_key_env_vars == ("AIVEN_AI_GATEWAY_KEY",)
    assert pdef.base_url_env_var == "AIVEN_AI_GATEWAY_BASE_URL"
    assert pdef.is_aggregator is False


def test_flat_model_alias_resolution(monkeypatch):
    import hermes_cli.model_switch as ms

    monkeypatch.setattr(
        ms,
        "list_provider_models",
        lambda _prov: ["claude-sonnet-5-5", "gpt-6.1-sol", "deepseek-v3.2"],
    )
    assert ms.resolve_alias("sonnet", "aiven") == (
        "aiven",
        "claude-sonnet-5-5",
        "sonnet",
    )


def test_registry_row_and_resolution(tmp_path, monkeypatch):
    monkeypatch.setenv("AIVEN_AI_GATEWAY_KEY", "aiven-test-key")
    monkeypatch.delenv("AIVEN_AI_GATEWAY_BASE_URL", raising=False)
    from hermes_cli.auth import (
        PROVIDER_REGISTRY,
        resolve_api_key_provider_credentials,
        resolve_provider,
    )

    row = PROVIDER_REGISTRY["aiven"]
    assert row.api_key_env_vars == ("AIVEN_AI_GATEWAY_KEY",)
    assert row.base_url_env_var == "AIVEN_AI_GATEWAY_BASE_URL"
    assert resolve_provider("aiven") == "aiven"
    creds = resolve_api_key_provider_credentials("aiven")
    assert creds["api_key"] == "aiven-test-key"
    assert creds["base_url"] == "https://ai.aiven.io/v1"


def test_fetch_models_reads_public_catalogue(monkeypatch):
    seen = []

    def fake_open(req, *, timeout, **_kwargs):
        seen.append(req)
        models = [
            {"id": "claude-sonnet-5-5"},
            {"id": "gpt-6.1-sol"},
            {"id": "deepseek-v3.2"},
        ]
        return io.BytesIO(json.dumps({"object": "list", "data": models}).encode())

    monkeypatch.setattr("hermes_cli.urllib_security.open_credentialed_url", fake_open)
    p = get_provider_profile("aiven")

    assert p.fetch_models(api_key="aiven-test-key", base_url=p.base_url) == [
        "claude-sonnet-5-5",
        "gpt-6.1-sol",
        "deepseek-v3.2",
    ]
    assert seen[0].full_url == "https://ai.aiven.io/v1/models"
    assert seen[0].get_header("Authorization") == "Bearer aiven-test-key"


def test_offline_models_have_declared_capabilities():
    profile = get_provider_profile("aiven")
    assert profile.fallback_models
    vision_models = set()
    text_only_models = set()
    for model in profile.fallback_models:
        facts = profile.model_capabilities[model]
        assert facts["supports_tools"] is True
        assert facts["context_window"] > 0
        if facts["supports_vision"]:
            vision_models.add(model)
        else:
            text_only_models.add(model)
    assert {
        "claude-haiku-4-5",
        "claude-sonnet-5-5",
        "gpt-6.1-sol",
        "qwen3-vl-235b",
    } <= vision_models
    assert {"deepseek-v3.2", "qwen3-coder-30b", "voxtral-mini-3b"} <= text_only_models


def test_generate_models_parses_v2_image_mime_modalities():
    import importlib.util

    script = Path(__file__).resolve().parents[1] / "tools" / "generate_models.py"
    spec = importlib.util.spec_from_file_location("aiven_generate_models", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    capabilities = module.capabilities

    snapshot = {
        "aiven": {
            "models": {
                "vision-chat": {
                    "type": "chat",
                    "capabilities": {
                        "tools": {"supported": True},
                        "reasoning": {"supported": True},
                    },
                    "modalities": {
                        "input": ["text", "image/jpeg", "image/png"],
                        "output": ["text"],
                    },
                    "limit": {"context": 200000},
                },
                "text-chat": {
                    "type": "chat",
                    "capabilities": {
                        "tools": {"supported": True},
                        "reasoning": {"supported": False},
                    },
                    "modalities": {
                        "input": ["text", "audio/wav"],
                        "output": ["text"],
                    },
                    "limit": {"context": 128000},
                },
                "no-tools": {
                    "type": "chat",
                    "capabilities": {
                        "tools": {"supported": False},
                        "reasoning": {"supported": False},
                    },
                    "modalities": {
                        "input": ["text", "image/png"],
                        "output": ["text"],
                    },
                    "limit": {"context": 131072},
                },
            }
        }
    }
    parsed = capabilities(snapshot)
    assert set(parsed) == {"vision-chat", "text-chat", "no-tools"}
    assert parsed["vision-chat"]["supports_vision"] is True
    assert parsed["text-chat"]["supports_vision"] is False
    assert parsed["no-tools"]["supports_tools"] is False


def test_per_model_vision_routing_and_tool_result_media_guard(monkeypatch):
    from agent.auxiliary_client import (
        reset_runtime_main,
        resolve_vision_provider_client,
        set_runtime_main,
    )
    from agent.image_routing import decide_image_input_mode
    from agent.models_dev import get_model_capabilities
    from tools.vision_tools import (
        _accepts_tool_result_images,
        _native_tool_result_images,
        _supports_media_in_tool_results,
    )

    monkeypatch.setenv("AIVEN_AI_GATEWAY_KEY", "aiven-test-key")
    monkeypatch.setenv("AIVEN_AI_GATEWAY_BASE_URL", "http://127.0.0.1:18464/v1")
    profile = get_provider_profile("aiven")
    assert profile.supports_vision is False
    assert profile.supports_vision_tool_messages is False

    for model in (
        "gpt-6.1-sol",
        "claude-sonnet-5-5",
        "claude-haiku-5-5",
        "qwen3-vl-235b",
    ):
        caps = get_model_capabilities("aiven", model)
        assert caps is not None and caps.supports_vision is True
        assert decide_image_input_mode("aiven", model, {}) == "native"
        assert _supports_media_in_tool_results("aiven", model) is False
        assert _accepts_tool_result_images("aiven", model, {}) is False
        assert _native_tool_result_images("aiven", model, {}) is False

    for model in ("qwen3-coder-30b", "deepseek-v3.2"):
        caps = get_model_capabilities("aiven", model)
        assert caps is not None and caps.supports_vision is False
        assert decide_image_input_mode("aiven", model, {}) == "text"
        assert _supports_media_in_tool_results("aiven", model) is False
        assert _accepts_tool_result_images("aiven", model, {}) is False

    for unknown_model in ("future-unlisted-vision-model", "f-secure-deepseek-v4-flash"):
        assert get_model_capabilities("aiven", unknown_model) is None
        assert decide_image_input_mode("aiven", unknown_model, {}) == "text"
        assert _accepts_tool_result_images("aiven", unknown_model, {}) is False

    token = set_runtime_main(
        "aiven",
        "gpt-6.1-sol",
        requested_provider="aiven",
        base_url="http://127.0.0.1:18464/v1",
        api_key="aiven-test-key",
        api_mode="chat_completions",
    )
    try:
        prov, client, resolved = resolve_vision_provider_client()
        assert prov == "aiven"
        assert client is not None
        assert resolved == "gpt-6.1-sol"
    finally:
        reset_runtime_main(token)

    token = set_runtime_main(
        "aiven",
        "qwen3-coder-30b",
        requested_provider="aiven",
        base_url="http://127.0.0.1:18464/v1",
        api_key="aiven-test-key",
        api_mode="chat_completions",
    )
    try:
        prov, client, resolved = resolve_vision_provider_client()
        assert client is None
        assert resolved is None
    finally:
        reset_runtime_main(token)


def test_auxiliary_routing_uses_main_model_not_hardcoded_haiku(monkeypatch):
    from agent.auxiliary_client import (
        _get_aux_model_for_provider,
        get_text_auxiliary_client,
        reset_runtime_main,
        resolve_provider_client,
        set_runtime_main,
    )
    from hermes_cli.config import save_config

    monkeypatch.setenv("AIVEN_AI_GATEWAY_KEY", "aiven-test-key")
    monkeypatch.setenv("AIVEN_AI_GATEWAY_BASE_URL", "http://127.0.0.1:18464/v1")
    profile = get_provider_profile("aiven")
    assert profile.default_aux_model == ""
    assert _get_aux_model_for_provider("aiven") == ""

    save_config({"model": {"provider": "aiven", "default": "qwen3-coder-30b"}})
    token = set_runtime_main(
        "aiven",
        "qwen3-coder-30b",
        requested_provider="aiven",
        base_url="http://127.0.0.1:18464/v1",
        api_key="aiven-test-key",
        api_mode="chat_completions",
    )
    try:
        client, aux_model = get_text_auxiliary_client("compression")
        assert client is not None
        assert aux_model == "qwen3-coder-30b"

        direct_client, direct_model = resolve_provider_client("aiven")
        assert direct_client is not None
        assert direct_model == "qwen3-coder-30b"
    finally:
        reset_runtime_main(token)


def test_context_length_honors_offline_snapshot_on_custom_base_url():
    from agent.model_metadata import get_model_context_length

    assert (
        get_model_context_length(
            "claude-haiku-5-5",
            base_url="http://127.0.0.1:18464/v1",
            api_key="aiven-test-key",
            provider="aiven",
        )
        == 1000000
    )
    assert (
        get_model_context_length(
            "qwen3-next-80b",
            base_url="http://127.0.0.1:18464/v1",
            api_key="aiven-test-key",
            provider="aiven",
        )
        == 262144
    )
    assert (
        get_model_context_length(
            "deepseek-v3.2",
            base_url="http://127.0.0.1:18464/v1",
            api_key="aiven-test-key",
            provider="aiven",
        )
        == 164000
    )
    assert (
        get_model_context_length(
            "deepseek-v3.2",
            base_url="http://127.0.0.1:18464/v1",
            api_key="aiven-test-key",
            config_context_length=96000,
            provider="aiven",
        )
        == 96000
    )


def test_the_payload_manifest_refuses_the_unsupported_release_floor(
    caplog, installed_plugin
):
    """Numeric release-floor checks and actual provider admission are separate assertions."""
    from hermes_cli.plugins_manifest import (
        parse_manifest_file,
        requires_hermes_error,
        version_satisfies,
    )

    plugin_dir = Path(__file__).resolve().parents[1]
    with caplog.at_level("WARNING", logger="hermes_cli.plugins"):
        manifest = parse_manifest_file(
            plugin_dir / "plugin.yaml", plugin_dir, "user", ""
        )
    assert not caplog.records, [r.getMessage() for r in caplog.records]
    assert manifest is not None
    assert manifest.requires_env == ["AIVEN_AI_GATEWAY_KEY"]
    assert manifest.capabilities == []
    assert manifest.requires_hermes == ">0.22.6"
    assert version_satisfies(manifest.requires_hermes, "0.22.6") is False
    assert version_satisfies(manifest.requires_hermes, "0.22.5") is False
    assert requires_hermes_error(manifest) is not None
    # Other tests can register runtime profiles; a cold process exercises admission itself.
    production_home = installed_plugin.parent / "production"
    installed_dir = production_home / "plugins/model-providers/aiven-provider"
    installed_dir.mkdir(parents=True)
    for filename in ("__init__.py", "models.json", "plugin.yaml"):
        shutil.copy2(plugin_dir / filename, installed_dir / filename)
    core = Path(providers.__file__).resolve().parent.parent
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from hermes_cli import plugins_manifest; "
            "plugins_manifest.running_hermes_version=lambda: '0.22.6'; "
            "import providers; "
            "assert providers.get_provider_profile('aiven') is None; "
            "assert 'requires hermes >0.22.6, running 0.22.6' "
            "in providers.plugin_load_error('aiven-provider')",
        ],
        cwd=core,
        env={
            "HOME": str(production_home),
            "HERMES_HOME": str(production_home),
            "PYTHONPATH": str(core),
            "PATH": os.environ["PATH"],
            "DO_NOT_TRACK": "1",
            "OTEL_SDK_DISABLED": "true",
        },
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_discovery_offers_new_live_ids_and_falls_back_offline(monkeypatch):
    from agent.models_dev import get_model_capabilities
    from hermes_cli.models import CuratedFallbackModels, provider_model_ids

    monkeypatch.setenv("AIVEN_AI_GATEWAY_KEY", "aiven-test-key")
    monkeypatch.setenv("AIVEN_AI_GATEWAY_BASE_URL", "http://127.0.0.1:18464/v1")

    def fake_open_ok(req, *, timeout, **_kwargs):
        assert req.full_url == "http://127.0.0.1:18464/v1/models"
        assert req.get_header("Authorization") == "Bearer aiven-test-key"
        return io.BytesIO(
            json.dumps(
                {
                    "object": "list",
                    "data": [
                        {"id": "future-unlisted-vision-model"},
                        {"id": "f-secure-deepseek-v4-flash"},
                        {"id": "gpt-6.1-sol"},
                    ],
                }
            ).encode()
        )

    monkeypatch.setattr(
        "hermes_cli.urllib_security.open_credentialed_url", fake_open_ok
    )
    discovered = provider_model_ids("aiven", force_refresh=True)
    assert not isinstance(discovered, CuratedFallbackModels)
    assert "future-unlisted-vision-model" in discovered
    assert "f-secure-deepseek-v4-flash" in discovered
    assert "gpt-6.1-sol" in discovered
    assert get_model_capabilities("aiven", "future-unlisted-vision-model") is None

    def fake_open_down(_req, *, timeout, **_kwargs):
        raise OSError("gateway unreachable")

    monkeypatch.setattr(
        "hermes_cli.urllib_security.open_credentialed_url", fake_open_down
    )
    offline = provider_model_ids("aiven", force_refresh=True)
    assert isinstance(offline, CuratedFallbackModels)
    assert tuple(offline) == get_provider_profile("aiven").fallback_models


def test_provider_is_scoped_to_its_installation(
    installed_plugin, tmp_path, monkeypatch
):
    result = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("profile_scope.py")),
            str(installed_plugin),
            str(Path(providers.__file__).resolve().parent.parent),
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_plugin_passes_hermes_admission_checks(installed_plugin):
    from hermes_cli.plugin_validate import validate_plugin_dir

    report = validate_plugin_dir(
        installed_plugin / "plugins" / "model-providers" / "aiven-provider"
    )
    assert report.ok, report.to_dict()


@pytest.mark.parametrize("model", ["claude-sonnet-5-5", "gpt-6.1-sol", "deepseek-v3.2"])
def test_native_client_tool_round_trip(model):
    from agent.auxiliary_client import resolve_provider_client
    from agent.transports.chat_completions import ChatCompletionsTransport

    requests = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append((self.path, self.headers.get("Authorization"), body))
            tool = len(requests) == 1
            message = (
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "lookup",
                                "arguments": '{"value":"42"}',
                            },
                        }
                    ],
                }
                if tool
                else {"role": "assistant", "content": "✓ résumé"}
            )
            payload = json.dumps(
                {
                    "id": "chatcmpl_aiven",
                    "object": "chat.completion",
                    "created": 0,
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "message": message,
                            "finish_reason": "tool_calls" if tool else "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 2,
                        "total_tokens": 12,
                    },
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    base_url = f"http://127.0.0.1:{server.server_port}/v1"
    client = None
    try:
        client, resolved = resolve_provider_client(
            "aiven",
            model=model,
            explicit_base_url=base_url,
            explicit_api_key="synthetic-key",
        )
        assert str(client.base_url).rstrip("/") == base_url
        assert resolved == model
        transport = ChatCompletionsTransport()
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "description": "Look up a value",
                    "parameters": {
                        "type": "object",
                        "properties": {"value": {"type": "string"}},
                    },
                },
            }
        ]
        messages = [{"role": "user", "content": "Look up 42"}]

        def send():
            kwargs = transport.build_kwargs(
                model=resolved,
                messages=messages,
                tools=tools,
                provider_profile=get_provider_profile("aiven"),
                max_tokens=None,
                max_tokens_param_fn=lambda _: {},
                timeout=5,
                reasoning_config=None,
                request_overrides=None,
                session_id="aiven-test",
                ollama_num_ctx=None,
                provider_name="aiven",
                base_url=base_url,
            )
            return client.chat.completions.create(**kwargs)

        first = send()
        normalized = transport.normalize_response(first)
        assert normalized.finish_reason == "tool_calls"
        assert first.choices[0].message.tool_calls[0].function.name == "lookup"
        assert json.loads(
            first.choices[0].message.tool_calls[0].function.arguments
        ) == {"value": "42"}
        messages.extend(
            [
                first.choices[0].message.model_dump(exclude_none=True),
                {"role": "tool", "tool_call_id": "call_1", "content": "42"},
            ]
        )
        second = transport.normalize_response(send())
        assert second.finish_reason == "stop"
        assert second.content == "✓ résumé"
        assert second.usage is not None
        assert second.usage.prompt_tokens == 10
        assert second.usage.completion_tokens == 2
        assert len(requests) == 2
        for route, auth, body in requests:
            assert route == "/v1/chat/completions"
            assert auth == "Bearer synthetic-key"
            assert body["model"] == model
        assert requests[1][2]["messages"][-1]["tool_call_id"] == "call_1"
    finally:
        if client is not None:
            client.close()
        server.shutdown()
        server.server_close()
        worker.join(timeout=5)
