import os
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM

MODELS = {
    "reprover": "facebook/reprover-byt5-small", # Encoder-Decoder for Lean
    "leancopilot": "lean-dojo/lean-ct5-small-new", # Used with leancopilot
    "theoremllama": "RickyDeSkywalker/TheoremLlama", # LLama-7B fine-tuned for lean
    "deepseek-prover": "deepseek-ai/deepseek-prover-v1.5-rl", # V2 is 671B; V1.5 is more probeable
    "goedel-prover": "Goedel-LM/Goedel-Prover-SFT",   # SOTA open prover
    "llama": "meta-llama/Llama-3.2-1B",               # Your primary general baseline
    "pythia": "EleutherAI/pythia-160m",               # Your small control
    "gpt2": "gpt2"
}

def download_models(model_dict, download_dir="./model_weights"):
    os.makedirs(download_dir, exist_ok=True)
    
    for name, repo_id in model_dict.items():
        print(f"--- Downloading {name} ({repo_id}) ---")
        try:
            snapshot_download(
                repo_id=repo_id,
                local_dir=os.path.join(download_dir, name),
                local_dir_use_symlinks=False
            )
        except Exception as e:
            print(f"Failed to download {name}: {e}")

def load_for_probing(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        output_hidden_states=True
    )
    return model, tokenizer

if __name__ == '__main__':
    download_models(MODELS)
    model, tokenizer = load_for_probing("./model_weights/pythia")
    print("Ready to extract layers!")