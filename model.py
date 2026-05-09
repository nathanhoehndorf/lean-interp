from __future__ import annotations
import os
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import HfFolder, snapshot_download
from transformer_lens import HookedTransformer, loading_from_pretrained as loading
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer, BitsAndBytesConfig

from config import ProbeConfig

HF_TOKEN_ENV_KEYS = ("HF_TOKEN", "HUGGINGFACEHUB_API_TOKEN")
ENV_PATH = Path(".env")
HF_CACHE_DIR = Path(".cache/hf_models")
GATED_MODELS = {
    "meta-llama/Llama-3.1-8B",
    "deepseek-ai/deepseek-math-7b-instruct",
}
MODEL_ALIASES = {
    "reprover": "kaiyuy/leandojo-lean4-retriever-tacgen-byt5-small",
    "reprover-tacgen": "kaiyuy/leandojo-lean4-tacgen-byt5-small",
    "reprover-retriever": "kaiyuy/leandojo-lean4-retriever-byt5-small",
    "reprover-retriever-tacgen": "kaiyuy/leandojo-lean4-retriever-tacgen-byt5-small",
}
SEQ2SEQ_MODEL_NAMES = {
    "kaiyuy/leandojo-lean4-tacgen-byt5-small",
    "kaiyuy/leandojo-lean4-retriever-byt5-small",
    "kaiyuy/leandojo-lean4-retriever-tacgen-byt5-small",
}


def get_hf_token() -> str | None:
    for env_key in HF_TOKEN_ENV_KEYS:
        token = os.environ.get(env_key)
        if token:
            return token
    if ENV_PATH.exists():
        try:
            with ENV_PATH.open("r", encoding="utf-8") as env_file:
                for line in env_file:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, _, value = line.partition("=")
                    if key.strip() in HF_TOKEN_ENV_KEYS:
                        token = value.strip().strip('"').strip("'")
                        if token:
                            return token
        except Exception:
            pass
    try:
        return HfFolder.get_token()
    except Exception:
        return None


def require_hf_token(model_name: str) -> str:
    token = get_hf_token()
    if model_name in GATED_MODELS and not token:
        raise EnvironmentError(
            "Gated model loading requires a Hugging Face auth token. "
            "Set HF_TOKEN or HUGGINGFACEHUB_API_TOKEN in your environment, "
            "or run `huggingface-cli login` before initialization."
        )
    return token


def get_device_map(device: str) -> Any:
    if device == "cpu":
        return "cpu"
    return "auto"


def build_quantization_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )


def is_official_model_name(model_name: str) -> bool:
    if Path(model_name).exists():
        return True
    try:
        loading.get_official_model_name(model_name)
        return True
    except ValueError:
        return False


def resolve_model_name(model_name: str) -> str:
    return MODEL_ALIASES.get(model_name, model_name)


def is_seq2seq_model_name(model_name: str) -> bool:
    return resolve_model_name(model_name) in SEQ2SEQ_MODEL_NAMES


def is_seq2seq_local_path(model_source: str) -> bool:
    path = Path(model_source)
    if not path.exists():
        return False
    try:
        cfg = AutoConfig.from_pretrained(model_source)
        return getattr(cfg, "is_encoder_decoder", False)
    except Exception:
        return False


def download_model_repo(model_name: str, token: str | None) -> Path:
    HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    local_path = Path(snapshot_download(
        repo_id=model_name,
        cache_dir=str(HF_CACHE_DIR),
        token=token,
        repo_type="model",
    ))
    return local_path


def load_tokenizer(config: ProbeConfig, model_source: str, token: str | None) -> Any:
    tokenizer_source = config.tokenizer_name or model_source
    return AutoTokenizer.from_pretrained(
        tokenizer_source,
        token=token,
    )


def load_hf_model(
    model_source: str,
    tokenizer: Any,
    device: str,
    trust_remote_code: bool,
    quantize: bool,
    seq2seq: bool,
) -> Any:
    hf_kwargs: dict[str, Any] = {
        "trust_remote_code": trust_remote_code,
    }
    if device == "cpu":
        hf_kwargs["dtype"] = torch.float32
    if not seq2seq and quantize and device != "cpu":
        hf_kwargs["quantization_config"] = build_quantization_config()

    if seq2seq:
        hf_model = AutoModelForSeq2SeqLM.from_pretrained(
            model_source,
            **hf_kwargs,
        )
    else:
        hf_model = AutoModelForCausalLM.from_pretrained(
            model_source,
            **hf_kwargs,
        )

    hf_model.tokenizer = tokenizer
    if device != "cpu":
        hf_model = hf_model.to(device)
    return hf_model


def load_local_model_from_hf(
    model_source: str,
    tokenizer: Any,
    device: str,
    trust_remote_code: bool,
    quantize: bool,
    token: str | None,
    seq2seq: bool,
) -> Any:
    if seq2seq:
        return load_hf_model(model_source, tokenizer, device, trust_remote_code, quantize, seq2seq=True)

    hf_model = AutoModelForCausalLM.from_pretrained(
        model_source,
        trust_remote_code=trust_remote_code,
        dtype=torch.float32,
    )
    hf_cfg = hf_model.config.to_dict()

    cfg = loading.get_pretrained_model_config(
        model_source,
        hf_cfg=hf_cfg,
        fold_ln=True,
        device=device,
        dtype=torch.float32,
        trust_remote_code=trust_remote_code,
    )

    state_dict = loading.get_pretrained_state_dict(
        model_source,
        cfg,
        hf_model,
        dtype=torch.float32,
        trust_remote_code=trust_remote_code,
    )

    model = HookedTransformer(
        cfg,
        tokenizer,
        move_to_device=False,
        default_padding_side="right",
    )
    model.load_and_process_state_dict(
        state_dict,
        fold_ln=True,
        center_writing_weights=True,
        center_unembed=True,
        fold_value_biases=True,
        refactor_factored_attn_matrices=False,
    )
    model.move_model_modules_to_device()
    return model


def load_model(config: ProbeConfig) -> Any:
    if config.needs_gated_auth:
        token = require_hf_token(config.model_name)
    else:
        token = get_hf_token()

    model_name = resolve_model_name(config.model_name)
    trust_remote_code = False
    quantize = config.quantize
    if config.quantize and config.device == "cpu":
        print("Warning: 4-bit quantization is not supported on CPU; loading standard precision instead.")
        quantize = False

    seq2seq_mode = is_seq2seq_model_name(model_name)

    if not is_official_model_name(model_name):
        print(f"Model {model_name} is not official for transformer-lens; downloading locally.")
        local_model_path = download_model_repo(model_name, token)
        model_source = str(local_model_path)
        trust_remote_code = True
    else:
        model_path = Path(model_name)
        model_source = str(model_path.resolve()) if model_path.exists() else model_name

    if not seq2seq_mode and Path(model_source).exists():
        seq2seq_mode = is_seq2seq_local_path(model_source)

    tokenizer = load_tokenizer(config, model_source, token)

    print(
        f"Loading {model_name} from {model_source} with device={config.device} "
        f"and {'4-bit quantization' if quantize else 'standard precision'}"
    )

    if seq2seq_mode:
        return load_hf_model(
            model_source,
            tokenizer,
            config.device,
            trust_remote_code,
            quantize,
            seq2seq=True,
        )

    if Path(model_source).exists():
        return load_local_model_from_hf(
            model_source,
            tokenizer,
            config.device,
            trust_remote_code,
            quantize,
            token,
            seq2seq=False,
        )

    load_kwargs: dict[str, Any] = {
        "device_map": get_device_map(config.device),
    }
    if quantize:
        load_kwargs["quantization_config"] = build_quantization_config()
    if trust_remote_code:
        load_kwargs["trust_remote_code"] = True

    return HookedTransformer.from_pretrained(
        model_source,
        device=config.device,
        tokenizer=tokenizer,
        **load_kwargs,
    )
