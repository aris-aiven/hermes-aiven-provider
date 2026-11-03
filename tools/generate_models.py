"""Generate the offline Hermes metadata from a models.dev v2 snapshot."""

import argparse
import json
from pathlib import Path


def _supports_vision(modalities: dict) -> bool:
    inputs = modalities.get("input", ())
    return any(
        isinstance(item, str) and (item == "image" or item.startswith("image/"))
        for item in inputs
    )


def capabilities(snapshot: dict) -> dict:
    # Non-tool chat models stay in with supports_tools=False: the account list names them
    # without capabilities, so this negative record is what keeps them out of the picker.
    entries = {}
    for model_id, model in snapshot["aiven"]["models"].items():
        if model["type"] != "chat":
            continue
        tools = model["capabilities"]["tools"]["supported"]
        if not isinstance(tools, bool):
            raise TypeError(f"Invalid tool declaration for {model_id}")
        context = model["limit"]["context"]
        if not isinstance(context, int) or isinstance(context, bool) or context <= 0:
            raise ValueError(f"Invalid context window for {model_id}")
        entries[model_id] = {
            "supports_tools": tools,
            "supports_reasoning": model["capabilities"]["reasoning"]["supported"],
            "supports_vision": _supports_vision(model["modalities"]),
            "context_window": context,
        }
    if not any(entry["supports_tools"] for entry in entries.values()):
        raise ValueError("The snapshot contains no tool-capable Aiven chat models")
    return entries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "snapshot", type=Path, help="Complete models.dev /api/v2.json snapshot"
    )
    parser.add_argument("--check", action="store_true", help="Compare without writing")
    args = parser.parse_args()
    entries = capabilities(json.loads(args.snapshot.read_text(encoding="utf-8")))
    target = Path(__file__).resolve().parents[1] / "models.json"
    if args.check:
        if json.loads(target.read_text(encoding="utf-8")) != entries:
            parser.exit(1, "Offline metadata differs from the snapshot\n")
    else:
        target.write_text(json.dumps(entries, indent=2) + "\n", encoding="utf-8")
    tool_models = sum(entry["supports_tools"] for entry in entries.values())
    print(f"{len(entries)} Aiven chat models, {tool_models} tool-capable")


if __name__ == "__main__":
    main()
