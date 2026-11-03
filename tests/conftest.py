"""Install the plugin into an isolated Hermes home for every test.

The installed copy gets a test-only numeric version identity. This avoids the canonical core
fixture's deliberately unknown version sentinel. These tests qualify development behavior;
real-runtime admission is tested separately and no released core support is claimed.
"""

import re
import shutil
from pathlib import Path

import pytest
from hermes_constants import reset_hermes_home_override, set_hermes_home_override


def development_manifest(payload: Path) -> str:
    from hermes_cli.plugins_manifest import running_hermes_version

    pin = f'requires_hermes: "=={running_hermes_version()}"  # test-only development build'
    text, count = re.subn(
        r"(?m)^requires_hermes: .*$", pin, payload.read_text(encoding="utf-8")
    )
    assert count == 1
    return text


@pytest.fixture(autouse=True)
def installed_plugin(tmp_path, monkeypatch):
    from hermes_cli import plugins_manifest

    monkeypatch.setattr(plugins_manifest, "running_hermes_version", lambda: "0.22.6")
    home = tmp_path / "home"
    plugin_dir = home / "plugins" / "model-providers" / "aiven-provider"
    source = Path(__file__).resolve().parents[1]
    plugin_dir.mkdir(parents=True)
    for filename in ("__init__.py", "models.json"):
        shutil.copy2(source / filename, plugin_dir / filename)
    (plugin_dir / "plugin.yaml").write_text(
        development_manifest(source / "plugin.yaml"), encoding="utf-8"
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: home)
    token = set_hermes_home_override(home)
    try:
        yield home
    finally:
        reset_hermes_home_override(token)
