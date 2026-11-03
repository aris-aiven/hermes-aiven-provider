"""Aiven AI Gateway provider profile."""

import json
from pathlib import Path

from providers import register_provider
from providers.base import ProviderProfile

models = json.loads(Path(__file__).with_name("models.json").read_text(encoding="utf-8"))


class AivenProfile(ProviderProfile):
    def get_model_context_length(self, model: str) -> int | None:
        entry = self.model_capabilities.get(model)
        ctx = entry.get("context_window") if isinstance(entry, dict) else None
        return (
            ctx
            if isinstance(ctx, int) and not isinstance(ctx, bool) and ctx > 0
            else None
        )

    def fetch_models(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 8.0,
    ) -> list[str] | None:
        """The account's model list without the models the catalogue declares tool-less.

        The list carries no capabilities, so IDs the catalogue does not know (private
        grants, new models) stay.
        """
        live = super().fetch_models(api_key=api_key, base_url=base_url, timeout=timeout)
        if live is None:
            return None
        return [
            model
            for model in live
            if self.model_capabilities.get(model, {}).get("supports_tools") is not False
        ]


aiven = AivenProfile(
    name="aiven",
    aliases=("aiven-ai-gateway", "aiven-gateway", "aiven_ai_gateway"),
    display_name="Aiven AI Gateway",
    description="Aiven AI Gateway (Claude, GPT, DeepSeek, Qwen, GLM via ai.aiven.io)",
    signup_url="https://console.aiven.io/",
    env_vars=("AIVEN_AI_GATEWAY_KEY", "AIVEN_AI_GATEWAY_BASE_URL"),
    base_url="https://ai.aiven.io/v1",
    supports_vision=False,
    supports_vision_tool_messages=False,
    hostnames=("ai.aiven.io",),
    models_dev_id="aiven",
    catalog_aliases=True,
    fallback_models=tuple(
        model for model, facts in models.items() if facts["supports_tools"]
    ),
    model_capabilities=models,
    authoritative_catalog=True,
    overlay_aliases=("aiven-ai-gateway", "aiven-gateway", "aiven_ai_gateway"),
    provider_overlay={
        "base_url_override": "https://ai.aiven.io/v1",
        "extra_env_vars": ("AIVEN_AI_GATEWAY_KEY",),
        "base_url_env_var": "AIVEN_AI_GATEWAY_BASE_URL",
    },
)

register_provider(aiven)
