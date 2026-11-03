"""The account's model list, not the bundled seed, decides which Aiven models are offered."""

import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import providers
import pytest
from providers import get_provider_profile

PRIVATE = "f-secure-deepseek-v4-flash"
REMOVED = "claude-haiku-4-5"


class Gateway:
    """Local ``/v1/models`` endpoint answering per bearer key: a model list or an HTTP status."""

    def __init__(self):
        self.accounts = {}
        self.hits = []
        gateway = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass

            def do_GET(self):
                key = (self.headers.get("Authorization") or "").removeprefix("Bearer ")
                gateway.hits.append((self.path, key))
                answer = gateway.accounts.get(key, 401)
                if isinstance(answer, int):
                    self.send_response(answer)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                payload = json.dumps({
                    "object": "list",
                    "data": [{"id": model, "object": "model"} for model in answer],
                }).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.worker = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.worker.start()
        self.base_url = f"http://127.0.0.1:{self.server.server_port}/v1"

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self.worker.join(timeout=5)


@pytest.fixture
def gateway():
    server = Gateway()
    try:
        yield server
    finally:
        server.close()


@pytest.fixture
def second_gateway():
    server = Gateway()
    try:
        yield server
    finally:
        server.close()


def use(monkeypatch, gateway, key):
    monkeypatch.setenv("AIVEN_AI_GATEWAY_KEY", key)
    monkeypatch.setenv("AIVEN_AI_GATEWAY_BASE_URL", gateway.base_url)


def seed():
    return list(get_provider_profile("aiven").fallback_models)


def picker(*, refresh=False):
    from hermes_cli.models import cached_provider_model_ids

    return cached_provider_model_ids("aiven", force_refresh=refresh)


def shown():
    """The rows the /model and GUI pickers list, with the curated row Hermes builds for a
    plugin profile (its fallback_models)."""
    from hermes_cli.model_switch_providers import _live_or_curated_ids

    return _live_or_curated_ids("aiven", {"aiven": seed()})


def setup_offer():
    """The list first-time `hermes model` setup offers for the configured key and endpoint."""
    from hermes_cli.auth import PROVIDER_REGISTRY
    from hermes_cli.model_setup_flows import _api_key_provider_model_list

    return _api_key_provider_model_list(
        "aiven",
        PROVIDER_REGISTRY["aiven"],
        os.environ["AIVEN_AI_GATEWAY_KEY"],
        "AIVEN_AI_GATEWAY_KEY",
        os.environ["AIVEN_AI_GATEWAY_BASE_URL"],
    )


def test_account_list_replaces_bundled_seed(monkeypatch, gateway):
    from hermes_cli.models import provider_model_ids

    gateway.accounts["key-a"] = ["gpt-5.5", PRIVATE]
    use(monkeypatch, gateway, "key-a")
    assert REMOVED in seed()

    assert provider_model_ids("aiven", force_refresh=True) == ["gpt-5.5", PRIVATE]
    assert picker(refresh=True) == ["gpt-5.5", PRIVATE]
    assert picker() == ["gpt-5.5", PRIVATE]


def test_known_non_tool_ids_are_hidden_and_unknown_private_ids_kept(
    monkeypatch, gateway
):
    from agent.models_dev import get_model_capabilities
    from hermes_cli.models import provider_model_ids

    gateway.accounts["key-a"] = ["mistral-7b", "gpt-5.5", "gemma-3-27b", PRIVATE]
    use(monkeypatch, gateway, "key-a")

    assert provider_model_ids("aiven", force_refresh=True) == ["gpt-5.5", PRIVATE]
    for model in ("mistral-7b", "gemma-3-27b"):
        assert model not in seed()
        caps = get_model_capabilities("aiven", model)
        assert caps is not None and caps.supports_tools is False
    assert get_model_capabilities("aiven", PRIVATE) is None
    assert setup_offer() == ["gpt-5.5", PRIVATE]
    assert picker(refresh=True) == ["gpt-5.5", PRIVATE]
    assert shown() == ["gpt-5.5", PRIVATE]


def test_empty_account_list_is_a_successful_answer(monkeypatch, gateway):
    from hermes_cli.models import CuratedFallbackModels, provider_model_ids

    gateway.accounts["key-a"] = []
    use(monkeypatch, gateway, "key-a")

    live = provider_model_ids("aiven", force_refresh=True)
    assert live == []
    assert not isinstance(live, CuratedFallbackModels)


def test_empty_account_list_clears_the_cached_catalogue(monkeypatch, gateway):
    gateway.accounts["key-a"] = ["gpt-5.5", PRIVATE]
    use(monkeypatch, gateway, "key-a")
    assert picker(refresh=True) == ["gpt-5.5", PRIVATE]

    gateway.accounts["key-a"] = []
    assert picker(refresh=True) == []
    assert picker() == []
    assert shown() == []
    assert setup_offer() == []


def test_failure_keeps_the_same_identity_catalogue(monkeypatch, gateway):
    from hermes_cli.models import CuratedFallbackModels, provider_model_ids

    gateway.accounts["key-a"] = ["gpt-5.5", PRIVATE]
    use(monkeypatch, gateway, "key-a")
    assert picker(refresh=True) == ["gpt-5.5", PRIVATE]

    gateway.accounts["key-a"] = 503
    offline = provider_model_ids("aiven", force_refresh=True)
    assert isinstance(offline, CuratedFallbackModels)
    assert list(offline) == seed()
    assert picker(refresh=True) == ["gpt-5.5", PRIVATE]


def test_a_failed_request_is_sent_once(monkeypatch, gateway):
    gateway.accounts["key-a"] = 503
    use(monkeypatch, gateway, "key-a")

    assert picker(refresh=True) == seed()
    assert len(gateway.hits) == 1
    assert setup_offer() == seed()
    assert len(gateway.hits) == 2


def test_failure_without_a_catalogue_offers_the_bundled_seed(monkeypatch, gateway):
    gateway.accounts["key-a"] = 503
    use(monkeypatch, gateway, "key-a")

    offered = picker(refresh=True)
    assert offered == seed()
    assert shown() == seed()
    assert REMOVED in offered and PRIVATE not in offered


def test_key_change_never_offers_the_previous_accounts_private_ids(
    monkeypatch, gateway
):
    gateway.accounts["key-a"] = ["gpt-5.5", PRIVATE]
    gateway.accounts["key-b"] = 401
    use(monkeypatch, gateway, "key-a")
    assert picker(refresh=True) == ["gpt-5.5", PRIVATE]

    use(monkeypatch, gateway, "key-b")
    assert PRIVATE not in picker()
    assert PRIVATE not in picker(refresh=True)

    gateway.accounts["key-b"] = ["claude-sonnet-5-5"]
    assert picker(refresh=True) == ["claude-sonnet-5-5"]


def test_endpoint_change_never_offers_the_previous_endpoints_private_ids(
    monkeypatch, gateway, second_gateway
):
    gateway.accounts["key-a"] = ["gpt-5.5", PRIVATE]
    second_gateway.accounts["key-a"] = 503
    use(monkeypatch, gateway, "key-a")
    assert picker(refresh=True) == ["gpt-5.5", PRIVATE]

    use(monkeypatch, second_gateway, "key-a")
    assert PRIVATE not in picker()
    assert picker(refresh=True) == seed()


def configure_relay(relay):
    from hermes_cli.config import save_config

    save_config({
        "model": {"provider": "aiven", "default": "gpt-5.5", "base_url": relay.base_url}
    })


def test_configured_endpoint_list_replaces_seed_and_hides_non_tool_ids(
    monkeypatch, gateway, second_gateway
):
    gateway.accounts["key-a"] = 503
    second_gateway.accounts["key-a"] = ["mistral-7b", "gpt-5.5", PRIVATE]
    use(monkeypatch, gateway, "key-a")
    configure_relay(second_gateway)

    assert picker(refresh=True) == ["gpt-5.5", PRIVATE]
    assert gateway.hits == []


def test_configured_endpoint_empty_list_is_a_successful_answer(
    monkeypatch, gateway, second_gateway
):
    from hermes_cli.models import provider_model_ids

    gateway.accounts["key-a"] = 503
    second_gateway.accounts["key-a"] = []
    use(monkeypatch, gateway, "key-a")
    configure_relay(second_gateway)

    assert provider_model_ids("aiven", force_refresh=True) == []


def test_configured_endpoint_change_never_offers_the_previous_endpoints_private_ids(
    monkeypatch, gateway, second_gateway
):
    gateway.accounts["key-a"] = 503
    second_gateway.accounts["key-a"] = ["gpt-5.5", PRIVATE]
    use(monkeypatch, gateway, "key-a")
    configure_relay(second_gateway)
    assert picker(refresh=True) == ["gpt-5.5", PRIVATE]

    from hermes_cli.config import save_config

    save_config({"model": {"provider": "aiven", "default": "gpt-5.5"}})
    assert PRIVATE not in picker()


def read_after_restart(home):
    env = {
        "PATH": os.environ.get("PATH", ""),
        "HOME": os.environ.get("HOME", ""),
        "HERMES_HOME": str(home),
        "AIVEN_AI_GATEWAY_KEY": os.environ["AIVEN_AI_GATEWAY_KEY"],
        "AIVEN_AI_GATEWAY_BASE_URL": os.environ["AIVEN_AI_GATEWAY_BASE_URL"],
    }
    result = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("catalogue_read.py")),
            str(Path(providers.__file__).resolve().parent.parent),
        ],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
        env=env,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_restart_serves_the_warm_catalogue_and_a_cold_home_the_seed(
    installed_plugin, monkeypatch, gateway
):
    gateway.accounts["key-a"] = ["gpt-5.5", PRIVATE]
    use(monkeypatch, gateway, "key-a")
    assert picker(refresh=True) == ["gpt-5.5", PRIVATE]
    gateway.accounts["key-a"] = 503
    hits = len(gateway.hits)

    assert read_after_restart(installed_plugin) == ["gpt-5.5", PRIVATE]
    assert len(gateway.hits) == hits

    (installed_plugin / "provider_models_cache.json").unlink()
    assert read_after_restart(installed_plugin) == seed()


def test_restart_serves_a_warm_empty_catalogue(installed_plugin, monkeypatch, gateway):
    gateway.accounts["key-a"] = []
    use(monkeypatch, gateway, "key-a")
    assert picker(refresh=True) == []
    hits = len(gateway.hits)

    assert read_after_restart(installed_plugin) == []
    assert len(gateway.hits) == hits
