"""Run outside pytest collection, which imports the plugin package's __init__."""

import sys
from pathlib import Path

sys.path.insert(0, sys.argv[2])

from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from providers import get_provider_profile, provider_source

home = Path(sys.argv[1])
token = set_hermes_home_override(home)
try:
    assert get_provider_profile("aiven") is not None
    assert provider_source("aiven") == "user"
    other = set_hermes_home_override(home.parent / "different-home")
    try:
        assert get_provider_profile("aiven") is None
    finally:
        reset_hermes_home_override(other)
finally:
    reset_hermes_home_override(token)
