# Aiven AI Gateway for Hermes

Local standalone model-provider plugin candidate. It registers Aiven through
Hermes's public provider-profile API and uses Hermes's existing Chat
Completions transport.

## Local setup

Create an AI Gateway key in the [Aiven Console](https://console.aiven.io/).
Copy `__init__.py`, `plugin.yaml` and `models.json` into
`~/.hermes/plugins/model-providers/aiven-provider/`, or the equivalent directory
under your configured `HERMES_HOME`.

```sh
export AIVEN_AI_GATEWAY_KEY="<your AI Gateway key>"
hermes plugins validate ~/.hermes/plugins/model-providers/aiven-provider
hermes --provider aiven --model claude-sonnet-5-5
```

`AIVEN_AI_GATEWAY_BASE_URL` optionally overrides `https://ai.aiven.io/v1`.
Hermes discovers the installed profile for provider selection, key resolution,
model aliases and authenticated model discovery. The offline snapshot records
68 Aiven chat models. The 63 tool-capable ones (38 with image input, 25
text/audio-only) form the bundled seed. The other five are kept with
`supports_tools: false`.

User image attachments route natively on models whose snapshot entry declares
`supports_vision: true` and fall back to text description on text-only or
undeclared models. Because the sampled Aiven Chat Completions route stripped `image_url`
parts from `role: "tool"` messages, the profile sets
`supports_vision_tool_messages=False` so `vision_analyze` sends images in a
`role: "user"` request rather than a tool-result attachment. `default_aux_model`
is left unset so auxiliary tasks use the session's configured main model.

## Model list

The profile sets `authoritative_catalog=True`. A successful `/models` answer
from the gateway is then the complete list that Hermes offers. The `/model`
picker, the GUI picker and first-time `hermes model` setup all list the
account's models in the order the gateway returns them. Hermes does not merge
them with the bundled seed or the models.dev registry. A model the account has
lost disappears from the list, even when the bundled seed still includes it.
An empty answer is a result too: Hermes caches it and lists no models instead
of the seed. Models the snapshot records as tool-less are hidden. Models the
snapshot does not know, such as private grants and new releases, are kept and
have no declared capabilities. The same rules apply when `model.base_url`
points Hermes at another endpoint.

Without a key, discovery sends no request. When a request fails, Hermes sends
one request and then falls back to its own cache. It offers the last list fetched
for the same
credentials and endpoint, or the bundled seed when no such list exists. The
same holds for an explicit refresh (`/model --refresh`, `hermes model
--refresh`) and for first-time setup with the key and endpoint already in
use. Hermes keys that cache by the key and endpoint that discovery actually
sends: the key resolved from the process environment, `$HERMES_HOME/.env`, a
request-scoped secret or the credential pool, `AIVEN_AI_GATEWAY_BASE_URL` and a
configured `model.base_url`. Changing any of them therefore never offers
another account's or endpoint's private models. A background refresh files
its answer under the credentials it was sent with.

The account list carries no capabilities. Newly discovered IDs still need
accurate metadata in models.dev and an offline snapshot refresh.

## Hermes core dependency

This behaviour needs a Hermes core change that no Hermes release contains yet:

- The `ProviderProfile.authoritative_catalog` field and the typed
  authoritative result that the picker, setup and disk cache carry.
- `model.base_url` added to the catalogue cache key for every provider.
- The background-refresh cache key taken before the request instead of after.
- The resolved key and endpoint in the cache key of an authoritative catalog.
- An explicit refresh that expires cached lists instead of deleting them.
- A failed setup probe that offers the cached list for the same key and
  endpoint.
- The key and endpoint captured once per discovery, so the request and the
  cache row it fills always name the same account.
- Native model-provider discovery enforcing the existing manifest version gate
  before importing a directory plugin, including hosted imports.

The core change is committed locally and unpublished. It is in the Hermes worktree
`aiven-ai-gateway-provider`, on top of `origin/main`
`353c9a914b9242a1dbeb0c33d6de3bd5ac0b1719` (local HEAD
`d73d0b08f7db0c72cbd0cf012ab072225e1bb4ff`), and it is not specific to Aiven.
Without it, constructing the profile fails with `TypeError:
ProviderProfile.__init__() got an unexpected keyword argument
'authoritative_catalog'`.

`plugin.yaml` declares `requires_hermes: ">0.22.6"`. The corrected local
loader enforces that requirement before import. Unmodified released cores
lack both the new profile field and this provider-loader correction; v0.22.6
attempts the import and reports the constructor `TypeError` above. The value
is not a support claim for a later release: the floor must be set to the first
release containing the required changes. Catalog admission remains pending
that merge and release. The shared native comparator's permissive policy for
an unknown runtime version is unchanged.

A development checkout with the change still reports the release it branched
from (`0.22.6`), so the corrected loader also rejects the production payload
there. The tests install a separate manifest and explicitly simulate the numeric
identity `0.22.6`, avoiding the canonical core fixture's `unknown` sentinel.
That fixture qualifies development behavior, not a release. Native subprocess
probes separately check the actual checkout's production refusal and development
loading. Neither test manifest is part of the payload.

## Maintenance and verification

Download a models.dev `/api/v2.json` snapshot, then regenerate offline metadata:

```sh
python tools/generate_models.py /path/to/api-v2.json
python tools/generate_models.py /path/to/api-v2.json --check
```

From a Hermes checkout, use its canonical isolated test runner:

```sh
scripts/run_tests.sh -j 1 /path/to/hermes-aiven-provider/tests/test_provider.py   /path/to/hermes-aiven-provider/tests/test_catalogue_reconciliation.py
```

The tests install into temporary homes and exercise discovery, key resolution,
aliases, per-model vision and auxiliary routing, context-window lookup, home
isolation, admission validation and local HTTP tool-result round trips for
Claude, GPT and DeepSeek. The reconciliation tests run a local `/models`
endpoint. They check the cached picker rows, the first-setup list and a fresh
process after a restart. They cover removed models, empty answers, failures
and their request count, key and endpoint changes, and tool-less and unknown
models.

Qualification is limited to Hermes revisions that carry the core change.
On the current candidate, 341 tests pass across 28 files, including the
93-case native admission grid and 33 standalone tests; all 14 repository
checks pass. The admission grid covers flat and nested payloads, both manifest
filenames, numeric constraints, malformed manifests, legacy directories,
unknown-version policy, bundled lookup and actual hosted imports. Separate
subprocess probes confirm production refusal and development loading using the
actual checkout version. These are local checks; they do not establish a
supporting release or independent acceptance of the final loader correction.

The preceding catalogue/picker selection passed 2,616 tests with 16 skips
across 171 files before the admission correction. Three existing expected
failures were reported separately. The standalone tests also passed on `origin/main`
`8d91e48efdfd68437781fe5a23f5cd1d991dce9e` with the change applied, using the
test-only manifest. An earlier version of the change, without the last four
items above, also passed them on release `v0.22.6`
(`f0d94882e89ab8d4098b5bb2f166adfb9ee0a0ab`) applied locally; that run was not
repeated for the current change. On unmodified `v0.22.6` and unmodified
`origin/main`, every test errored at the plugin import in that earlier run. Live gateway journeys
on the release used an earlier 57-model payload and were not repeated for this
one. Repository owner, publication location and immutable catalog pin remain
unestablished. An external plugin adds an installation step; the AIG-464
built-in acceptance requirement still needs a separate decision.
