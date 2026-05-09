import json
import random
from pathlib import Path
from typing import Any

from features import FEATURES


DEFAULT_INPUT = Path("data/states.jsonl")
DEFAULT_OUTPUT = Path("data/states_gold_dataset.json")


def extract_state(record: dict[str, Any]) -> str:
    return (
        record.get("state_before")
        or record.get("state")
        or record.get("goal")
        or record.get("source")
        or ""
    )


def build_labels(record: dict[str, Any], state: str) -> dict[str, Any]:
    labels = record.get("labels")
    if not isinstance(labels, dict):
        labels = {feature.name: feature.extract(state) for feature in FEATURES}
    else:
        for feature in FEATURES:
            if feature.name not in labels:
                labels[feature.name] = feature.extract(state)
    return labels


def normalize_record(record: dict[str, Any]) -> dict[str, Any]:
    state = extract_state(record)
    if not state:
        raise ValueError("JSONL record is missing a state_before/state/goal/source field")

    normalized = {
        "state": state,
        "labels": build_labels(record, state),
    }

    if "tactic" in record:
        normalized["tactic"] = record["tactic"]
    elif "tactic_name" in record:
        normalized["tactic"] = record["tactic_name"]

    if "domain" in record:
        normalized["domain"] = record["domain"]

    return normalized


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_json(data: dict[str, list[dict[str, Any]]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def split_dataset(records: list[dict[str, Any]], seed: int = 42) -> dict[str, list[dict[str, Any]]]:
    random.seed(seed)
    random.shuffle(records)
    n = len(records)
    tr_end = int(n * 0.7)
    val_end = int(n * 0.85)
    return {
        "train": records[:tr_end],
        "val": records[tr_end:val_end],
        "test": records[val_end:],
    }


def main(input_path: Path = DEFAULT_INPUT, output_path: Path = DEFAULT_OUTPUT) -> None:
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    raw_records = load_jsonl(input_path)
    normalized = [normalize_record(record) for record in raw_records]
    dataset = split_dataset(normalized)
    write_json(dataset, output_path)
    print(f"Wrote {len(normalized)} records to {output_path}")
    for split_name in ("train", "val", "test"):
        print(f"  {split_name}: {len(dataset[split_name])}")


if __name__ == "__main__":
    main()
