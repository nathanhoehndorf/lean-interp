import os
import json
import random
from lean_dojo_v2.lean_dojo import LeanGitRepo, trace
import re
from features import FEATURES


def extract_goal_depth(state: str) -> int:
    if "⊢" not in state:
        return 0
    goal = state.split("⊢")[1].strip()
    depth = goal.count('→') + goal.count('∀') + goal.count('∃')
    return depth

def extract_quantifier_presence(state: str) -> int:
    return 1 if '∀' in state or '∃' in state else 0

def extract_context_size(state: str) -> int:
    if "⊢" not in state:
        return 0
    context = state.split("⊢")[0].strip()
    return len([line for line in context.split('\n') if line.strip()])

def extract_type_complexity(state: str) -> int:
    return state.count('Type') + state.count('Sort') + state.count('Prop')

def extract_inductive_vs_direct(state: str) -> int:
    return 1 if 'inductive' in state.lower() or 'cases' in state.lower() else 0

def extract_equality_vs_inequality(state: str) -> int:
    if "⊢" not in state:
        return 0
    goal = state.split("⊢")[1]
    return 1 if '=' in goal else 0

def extract_unbound_variables(state: str) -> int:
    if "⊢" not in state:
        return 0
    context = state.split("⊢")[0]
    goal = state.split("⊢")[1]
    context_vars = set(re.findall(r'\b[a-z_][a-z0-9_]*\b', context))
    goal_vars = set(re.findall(r'\b[a-z_][a-z0-9_]*\b', goal))
    unbound = goal_vars - context_vars
    return len(unbound)

def extract_token_labels(state: str) -> dict:
    tokens = re.findall(r'\S+', state)
    is_var = []
    is_kwd = []
    keywords = {'Sort', 'Prop', 'Type', '∀', '∃', '⊢', '→', '∧', '∨', '¬', 'λ', 'let', 'in', 'if', 'then', 'else', 'match', 'with', 'cases', 'inductive', 'structure', 'def', 'theorem', 'lemma', 'axiom', 'variable', 'parameters', 'assume', 'have', 'show', 'by', 'exact', 'apply', 'rewrite', 'simp', 'intro', 'intros', 'revert', 'clear', 'cases', 'induction', 'contradiction', 'trivial', 'sorry'}
    for token in tokens:
        is_var.append(1 if re.match(r'^[a-z_][a-z0-9_]*$', token) and token not in keywords else 0)
        is_kwd.append(1 if token in keywords else 0)
    return {'tokens': tokens, 'is_var': is_var, 'is_kwd': is_kwd}


OUTPUT_PATH = "data/gold_dataset.json"
# Updated for std4 repository paths
# Update your TARGET_DOMAINS to match the Std library structure
TARGET_DOMAINS = {
    # Original Domains
    "Data Structures": "Std/Data",
    "Logic": "Std/Logic",
    "Time": "Std/Time",
    "Control": "Std/Control",
    "Classes": "Std/Classes",
    
    # System & I/O
    "FileSystem": "Std/IO/FS",
    "Network": "Std/IO/Net",
    "Concurrency": "Std/Async",
    "Environment": "Std/Sys/Env",
    
    # Mathematical & Scientific
    "Math": "Std/Math/Core",
    "Statistics": "Std/Math/Stats",
    "Geometry": "Std/Math/Geo",
    
    # Text & Encoding
    "String": "Std/Text/String",
    "Regex": "Std/Text/Regex",
    "Encoding": "Std/Text/Codec",
    
    # Security & Validation
    "Crypto": "Std/Sec/Crypto",
    "Auth": "Std/Sec/Auth",
    "Validation": "Std/Util/Valid",
    
    # Diagnostics
    "Logging": "Std/Diag/Log",
    "Testing": "Std/Diag/Test"
}

def load_existing_data():
    if os.path.exists(OUTPUT_PATH):
        with open(OUTPUT_PATH, "r") as f:
            return json.load(f)
    return {"train": [], "val": [], "test": []}

def save_data(data):
    with open(OUTPUT_PATH, "w") as f:
        json.dump(data, f, indent=2)


repo = LeanGitRepo(
    "https://github.com/leanprover/std4", 
    "main"
)

traced_repo = trace(repo)

# 1. Process Domain by Domain
for domain_name, path_prefix in TARGET_DOMAINS.items():
    print(f"\n--- Starting Section: {domain_name} ---")
    
    # Load what we have so far
    current_dataset = load_existing_data()
    
    # Use a set for fast lookup of theorems already processed (optional but recommended)
    # This requires storing theorem names, or just checking the data count
    domain_samples = []
    count = 0

    for file in traced_repo.traced_files:
        # IGNORE PATH FILTERS ENTIRELY FOR DEBUGGING
        theorems = file.get_traced_theorems()
        for thm in theorems:
            tactics = thm.get_traced_tactics()
            for tactic in tactics:
                state = tactic.state_before
                if state:
                    labels = {feature.name: feature.extract(state) for feature in FEATURES}
                    labels.update(extract_token_labels(state))
                    domain_samples.append({"state": state, "labels": labels, "domain": domain_name})
                    count += 1
    # 2. Split the NEW domain samples
    random.seed(42)
    random.shuffle(domain_samples)
    
    n = len(domain_samples)
    tr_end = int(n * 0.7)
    val_end = int(n * 0.85)

    # 3. Append to existing dataset
    current_dataset["train"].extend(domain_samples[:tr_end])
    current_dataset["val"].extend(domain_samples[tr_end:val_end])
    current_dataset["test"].extend(domain_samples[val_end:])

    # 4. Save immediately after finishing a domain
    save_data(current_dataset)
    print(f"Finished {domain_name}. Total samples now: {len(current_dataset['train'])}")

print("\nAll domains processed and saved to gold_dataset.json")