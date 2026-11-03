"""List the Aiven picker rows in a fresh process, as a restarted Hermes would."""

import json
import sys

sys.path.insert(0, sys.argv[1])

from hermes_cli.model_switch_providers import _live_or_curated_ids
from providers import get_provider_profile

seed = list(get_provider_profile("aiven").fallback_models)
print(json.dumps(_live_or_curated_ids("aiven", {"aiven": seed})))
