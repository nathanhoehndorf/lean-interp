import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pyarrow.parquet as pq

from features import FEATURES


def parse_traced_tactics(value: Any) -> List[Dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return []
    return []


def extract_state_records(row: Dict[str, Any], row_id: int) -> Iterable[Dict[str, Any]]:
    traced_tactics = parse_traced_tactics(row.get("traced_tactics"))
    for tactic_index, tactic in enumerate(traced_tactics):
        state_before = tactic.get("state_before") or tactic.get("state")
        if not state_before:
            continue

        entry = {
            "id": f"{row_id}-{tactic_index}",
            "url": row.get("url"),
            "commit": row.get("commit"),
            "file_path": row.get("file_path"),
            "full_name": row.get("full_name"),
            "state_before": state_before,
            "tactic": tactic.get("tactic") or tactic.get("annotated_tactic"),
        }
        yield entry


def label_record(entry: Dict[str, Any]) -> Dict[str, Any]:
    state = entry["state_before"]
    labels = {feature.name: feature.extract(state) for feature in FEATURES}
    return {**entry, "labels": labels}


def convert_parquet_files(parquet_dir: Path, output_path: Path, label: bool = False, limit: Optional[int] = None) -> None:
    parquet_files = sorted(parquet_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {parquet_dir}")

    count = 0
    with output_path.open("w", encoding="utf-8") as out_f:
        for parquet_file in parquet_files:
            reader = pq.ParquetFile(parquet_file)
            for batch in reader.iter_batches(batch_size=256):
                table = batch.to_pydict()
                n = len(next(iter(table.values()), []))
                for i in range(n):
                    if limit is not None and count >= limit:
                        return
                    row = {key: table[key][i] for key in table}
                    for entry in extract_state_records(row, count):
                        if limit is not None and count >= limit:
                            return
                        if label:
                            entry = label_record(entry)
                        out_f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                        count += 1

    print(f"Wrote {count} records to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert LeanDojo parquet exports into states.jsonl")
    parser.add_argument("--parquet-dir", type=str, default="data/data", help="Directory containing parquet files.")
    parser.add_argument("--output", type=str, default="states.jsonl", help="Output JSONL file path.")
    parser.add_argument("--label", action="store_true", help="Also compute feature labels and include them in the output.")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of tactic states to export.")
    args = parser.parse_args()

    convert_parquet_files(Path(args.parquet_dir), Path(args.output), label=args.label, limit=args.limit)


if __name__ == "__main__":
    main()
