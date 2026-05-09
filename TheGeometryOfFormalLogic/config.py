from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
import argparse
import torch


class Mode(str, Enum):
    SMOKE = "smoke"
    CONTROL = "control"
    TESTER = "tester"
    PRODUCTION = "production"


MODE_DEFAULTS = {
    Mode.SMOKE: {
        "model_name": "EleutherAI/pythia-70m",
        "sample_size": 50,
        "layer": 3,
    },
    Mode.CONTROL: {
        "model_name": "gpt2",
        "sample_size": 200,
        "layer": 0,
    },
    Mode.TESTER: {
        "model_name": "reprover-tacgen",
        "sample_size": 500,
        "layer": 0,
    },
    Mode.PRODUCTION: {
        "model_name": "reprover",
        "sample_size": 700,
        "layer": 0,
    },
}


@dataclass
class ProbeConfig:
    mode: Mode
    model_name: str
    dataset_path: str
    sample_size: int | None
    layer: int
    device: str
    sweep_layers: bool
    max_layer: int | None
    output_dir: str

    @staticmethod
    def from_args() -> "ProbeConfig":
        parser = argparse.ArgumentParser(description="Lean probing configuration")
        parser.add_argument(
            "--mode",
            choices=[mode.value for mode in Mode],
            default=Mode.SMOKE.value,
            help="Choose smoke, control, tester, or production mode.",
        )
        parser.add_argument(
            "--model-name",
            type=str,
            default=None,
            help="Optional override for the model name or local path.",
        )
        parser.add_argument(
            "--dataset-path",
            type=str,
            default="data/gold_dataset.json",
            help="Path to the dataset JSON file.",
        )
        parser.add_argument(
            "--sample-size",
            type=int,
            default=None,
            help="Optional sample size for the dataset subset used for probing.",
        )
        parser.add_argument(
            "--layer",
            type=int,
            default=None,
            help="Optional single layer index override for the probe.",
        )
        parser.add_argument(
            "--sweep-layers",
            action="store_true",
            help="Probe every available transformer layer instead of a single fixed layer.",
        )
        parser.add_argument(
            "--max-layer",
            type=int,
            default=None,
            help="Optional maximum layer index to include when sweeping layers.",
        )
        parser.add_argument(
            "--output-dir",
            type=str,
            default="data/results",
            help="Directory where figures and result files are written.",
        )
        parser.add_argument(
            "--device",
            type=str,
            default="cuda" if torch.cuda.is_available() else "cpu",
            help="Device for model execution.",
        )

        args = parser.parse_args()
        mode = Mode(args.mode)
        defaults = MODE_DEFAULTS[mode]

        model_name = args.model_name if args.model_name else defaults["model_name"]
        sample_size = args.sample_size if args.sample_size is not None else defaults["sample_size"]
        layer = args.layer if args.layer is not None else defaults["layer"]

        return ProbeConfig(
            mode=mode,
            model_name=model_name,
            dataset_path=args.dataset_path,
            sample_size=sample_size,
            layer=layer,
            device=args.device,
            sweep_layers=args.sweep_layers,
            max_layer=args.max_layer,
            output_dir=args.output_dir,
        )

    @property
    def quantize(self) -> bool:
        return self.mode in {Mode.CONTROL, Mode.TESTER, Mode.PRODUCTION} and self.device != "cpu"

    @property
    def needs_gated_auth(self) -> bool:
        return self.model_name in {
            "meta-llama/Meta-Llama-3-8B",
            "deepseek-ai/deepseek-math-7b-instruct",
        }

    @property
    def tokenizer_name(self) -> str | None:
        if self.model_name in {
            "meta-llama/Meta-Llama-3-8B",
            "deepseek-ai/deepseek-math-7b-instruct",
        }:
            return self.model_name
        return None

    def __str__(self) -> str:
        return (
            f"ProbeConfig(mode={self.mode}, model_name={self.model_name}, "
            f"dataset_path={self.dataset_path}, sample_size={self.sample_size}, "
            f"layer={self.layer}, sweep_layers={self.sweep_layers}, "
            f"max_layer={self.max_layer}, device={self.device}, output_dir={self.output_dir})"
        )
