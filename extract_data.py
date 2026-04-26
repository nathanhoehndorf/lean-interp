import os
import json
import random
from lean_dojo_v2.lean_dojo import LeanGitRepo, trace
import re

def extract_variable_count(state_before: str) -> int:
    if not state_before or "⊢" not in state_before:
        return 0
    
    local_context = state_before.split("⊢")[0].strip()
    if not local_context:
        return 0
    
    count = 0
    lines = local_context.split('\n')
    for line in lines:
        line = line.strip()
        if not line or line.startswith('case'):
            continue

        if ':' in line:
            names_part = line.split(':')[0].strip()
            clean_names = re.sub(r'[\{\}\[\]]', '', names_part).strip()
            vars_on_line = clean_names.split()
            count += len(vars_on_line)

    return count


OUTPUT_PATH = "gold_dataset.json"
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
                    y = extract_variable_count(state)
                    domain_samples.append({"state": state, "label": y, "domain": domain_name})
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