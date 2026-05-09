import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable

from features import FEATURES


def extract_state(record: Dict[str, Any]) -> str:
    return (
        record.get("state_before")
        or record.get("state")
        or record.get("goal")
        or record.get("source")
        or ""
    )


def build_labels(state: str) -> Dict[str, float]:
    return {feature.name: feature.extract(state) for feature in FEATURES}


def process_record(record: Dict[str, Any]) -> Dict[str, Any]:
    state = extract_state(record)
    if not state:
        raise ValueError("Record missing a Lean state field: state_before/state/goal/source")

    labels = record.get("labels")
    if labels is None:
        labels = build_labels(state)
    elif not isinstance(labels, dict):
        labels = build_labels(state)
    else:
        for feature in FEATURES:
            if feature.name not in labels:
                labels[feature.name] = feature.extract(state)

    output = {
        **record,
        "state": state,
        "state_before": state,
        "labels": labels,
    }
    return output


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def write_jsonl(records: Iterable[Dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Label LeanDojo JSONL states for probing features.")
    parser.add_argument("input_path", type=str, help="Input LeanDojo JSONL file.")
    parser.add_argument("output_path", type=str, help="Output labeled JSONL file.")
    parser.add_argument("--force", action="store_true", help="Recompute labels even if they already exist.")
    args = parser.parse_args()

    input_path = Path(args.input_path)
    output_path = Path(args.output_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    records = []
    for record in read_jsonl(input_path):
        if args.force or record.get("labels") is None:
            record = process_record(record)
        else:
            record = process_record(record)
        records.append(record)

    write_jsonl(records, output_path)
    print(f"Saved labeled JSONL to {output_path}")


if __name__ == "__main__":
    main()
