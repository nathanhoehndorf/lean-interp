import json
import os
import random
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM

from features import FEATURES

MODELS = {
    "reprover": "kaiyuy/leandojo-lean4-retriever-tacgen-byt5-small",
    "reprover-tacgen": "kaiyuy/leandojo-lean4-tacgen-byt5-small",
    "reprover-retriever": "kaiyuy/leandojo-lean4-retriever-byt5-small",
    "leancopilot": "lean-dojo/lean-ct5-small-new",
    "theoremllama": "RickyDeSkywalker/TheoremLlama",
    "deepseek-prover": "deepseek-ai/deepseek-prover-v1.5-rl",
    "goedel-prover": "Goedel-LM/Goedel-Prover-SFT",
    "llama": "meta-llama/Llama-3.2-1B",
    "pythia": "EleutherAI/pythia-160m",
    "gpt2": "gpt2",
}


class BaseLoader(ABC):
    @abstractmethod
    def load_prompts(self) -> Tuple[List[str], List[Dict[str, Any]], List[str], List[Dict[str, Any]]]:
        ...


class LeanDojoLoader(BaseLoader):
    def __init__(self, file_path: str, limit: Optional[int] = None, seed: int = 42):
        self.path = Path(file_path)
        self.limit = limit
        self.seed = seed

    def load_prompts(self) -> Tuple[List[str], List[Dict[str, Any]], List[str], List[Dict[str, Any]]]:
        records = self._load_or_build_labeled_records()

        if self.limit is not None and self.limit > 0 and len(records) > self.limit:
            random.seed(self.seed)
            records = random.sample(records, self.limit)

        random.seed(self.seed)
        random.shuffle(records)
        n = len(records)
        tr_end = int(n * 0.7)
        val_end = int(n * 0.85)

        train_records = records[:tr_end]
        test_records = records[val_end:]

        train_prompts = [r["state"] for r in train_records]
        train_labels = [r["labels"] for r in train_records]
        test_prompts = [r["state"] for r in test_records]
        test_labels = [r["labels"] for r in test_records]

        return train_prompts, train_labels, test_prompts, test_labels

    def _load_or_build_labeled_records(self) -> List[Dict[str, Any]]:
        labeled_path = self.path.with_name(self.path.stem + ".labeled").with_suffix(".jsonl")
        if labeled_path.exists():
            return list(self._read_jsonl(labeled_path))

        records = [self._normalize_record(r) for r in self._read_jsonl(self.path)]
        self._write_jsonl(records, labeled_path)
        return records

    def _read_jsonl(self, path: Path) -> Iterable[Dict[str, Any]]:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)

    def _write_jsonl(self, records: List[Dict[str, Any]], path: Path) -> None:
        with path.open("w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _normalize_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        state = self._extract_state(record)
        if not state:
            raise ValueError("LeanDojo JSONL record is missing a state_before field")

        labels = record.get("labels")
        if labels is None or not isinstance(labels, dict):
            labels = self._compute_labels(state)
        else:
            for feature in FEATURES:
                if feature.name not in labels:
                    labels[feature.name] = feature.extract(state)

        normalized: Dict[str, Any] = {
            "state": state,
            "state_before": state,
            "labels": labels,
        }

        if "tactic" in record:
            normalized["tactic"] = record["tactic"]
        elif "tactic_name" in record:
            normalized["tactic"] = record["tactic_name"]

        return normalized

    def _extract_state(self, record: Dict[str, Any]) -> str:
        return (
            record.get("state_before")
            or record.get("state")
            or record.get("goal")
            or record.get("source")
            or ""
        )

    def _compute_labels(self, state: str) -> Dict[str, float]:
        return {feature.name: feature.extract(state) for feature in FEATURES}


def download_models(model_dict, download_dir="./model_weights"):
    os.makedirs(download_dir, exist_ok=True)

    for name, repo_id in model_dict.items():
        print(f"--- Downloading {name} ({repo_id}) ---")
        try:
            snapshot_download(
                repo_id=repo_id,
                local_dir=os.path.join(download_dir, name),
                local_dir_use_symlinks=False,
            )
        except Exception as e:
            print(f"Failed to download {name}: {e}")


def load_for_probing(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        output_hidden_states=True,
    )
    return model, tokenizer


def load_prompts(file_path: str, sample_size: Optional[int] = None):
    if file_path.endswith(".jsonl"):
        loader = LeanDojoLoader(file_path, limit=sample_size)
        return loader.load_prompts()

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    train_prompts = [item["state"] for item in data.get("train", [])]
    train_labels = [item.get("labels", {"variable_count": item.get("label", 0), "is_var": [], "is_kwd": []}) for item in data.get("train", [])]

    test_prompts = [item["state"] for item in data.get("test", [])]
    test_labels = [item.get("labels", {"variable_count": item.get("label", 0), "is_var": [], "is_kwd": []}) for item in data.get("test", [])]

    if sample_size is None or sample_size <= 0:
        return train_prompts, train_labels, test_prompts, test_labels

    combined = list(zip(train_prompts, train_labels)) + list(zip(test_prompts, test_labels))
    random.seed(42)
    random.shuffle(combined)
    selected = combined[:sample_size]
    split = max(1, int(len(selected) * 0.8))
    train_selected = selected[:split]
    test_selected = selected[split:]

    train_prompts, train_labels = zip(*train_selected)
    test_prompts, test_labels = zip(*test_selected)

    return list(train_prompts), list(train_labels), list(test_prompts), list(test_labels)


if __name__ == '__main__':
    download_models(MODELS)
    model, tokenizer = load_for_probing("./model_weights/pythia")
    print("Ready to extract layers!")
