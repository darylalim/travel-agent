"""Upload the evaluation datasets in `evals/datasets/` to LangSmith.

    uv run python evals/upload.py              # create every dataset not already there
    uv run python evals/upload.py --dry-run    # validate only; no network, no key
    uv run python evals/upload.py --only trajectory

Each JSON file holds one dataset: `name`, `description` and `examples`, where
an example is `key`, `inputs`, `outputs` and `metadata`. Every `inputs` is the
graph's own input shape, `{"messages": [...]}`, so an evaluation target can pass
it straight to `agent.invoke`.

A dataset that already exists in LangSmith is **skipped, never overwritten**.
Its experiments are compared against the examples they ran on, so replacing them
in place would quietly change what every past score meant. To change one, upload
it under a new name, or delete the old one in LangSmith first.

Trip dates are literals and the search tools reject a date before today, so an
example goes stale on the day its dates pass. A run against it would still score
something, just no longer the behaviour it was written for. The upload refuses
those, apart from examples marked `dates_intentionally_past`, which are *about*
past dates. Move the dates forward rather than lifting the check.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import date
from pathlib import Path

from dotenv import load_dotenv

DATASETS_DIR = Path(__file__).parent / "datasets"
_ISO_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")


def load_datasets(directory: Path = DATASETS_DIR) -> dict[str, dict]:
    """Every dataset file in `directory`, keyed by file stem."""
    return {
        path.stem: json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(directory.glob("*.json"))
    }


def stale_dates(example: dict, today: date) -> list[str]:
    """ISO dates anywhere in an example's inputs or outputs before `today`.

    Outputs count too: a reference search argument dated in the past is a call
    the real tool rejects. Empty when the example is marked
    `dates_intentionally_past`.
    """
    if example.get("metadata", {}).get("dates_intentionally_past"):
        return []
    text = json.dumps([example["inputs"], example.get("outputs")])
    return sorted({d for d in _ISO_DATE.findall(text) if date.fromisoformat(d) < today})


def _check(datasets: dict[str, dict], today: date) -> list[str]:
    problems = []
    for stem, dataset in datasets.items():
        for example in dataset["examples"]:
            if stale := stale_dates(example, today):
                problems.append(f"{stem}/{example['key']}: dates already passed: {stale}")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dry-run", action="store_true", help="Validate only; upload nothing.")
    parser.add_argument("--only", action="append", help="Dataset file stem; repeatable.")
    args = parser.parse_args(argv)

    datasets = load_datasets()
    if args.only:
        unknown = set(args.only) - set(datasets)
        if unknown:
            parser.error(f"unknown dataset(s) {sorted(unknown)}; choose from {sorted(datasets)}")
        datasets = {stem: datasets[stem] for stem in args.only}

    # Imported here rather than at module scope so the test suite can import
    # this file without pulling in the travel_agent package.
    from travel_agent.clock import today

    if problems := _check(datasets, today()):
        print("Refusing to upload stale examples:", *problems, sep="\n  ", file=sys.stderr)
        return 1

    for stem, dataset in datasets.items():
        print(f"{stem}: {len(dataset['examples'])} examples, valid -> {dataset['name']!r}")
    if args.dry_run:
        return 0

    load_dotenv()
    from langsmith import Client

    client = Client()
    for stem, dataset in datasets.items():
        name = dataset["name"]
        if client.has_dataset(dataset_name=name):
            print(f"skip {name!r}: already exists in LangSmith")
            continue
        created = client.create_dataset(
            name,
            description=dataset["description"],
            metadata={"source_file": f"evals/datasets/{stem}.json"},
        )
        client.create_examples(
            dataset_id=created.id,
            examples=[
                {
                    "inputs": example["inputs"],
                    "outputs": example["outputs"],
                    "metadata": {**example.get("metadata", {}), "key": example["key"]},
                }
                for example in dataset["examples"]
            ],
        )
        print(f"created {name!r} with {len(dataset['examples'])} examples")
    return 0


if __name__ == "__main__":
    sys.exit(main())
