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

TARGET_DOMAINS = {
    "Algebra": "Mathlib/Algebra/",
    "Linear Algebra": "Mathlib/LinearAlgebra/",
    "Analysis": "Mathlib/Analysis/",
    "Probability": "Mathlib/Probability/",
    "Geometry": "Mathlib/Geometry/",
    "Logic/Set Theory": "Mathlib/Data"
}

repo = LeanGitRepo(
    "https://github.com/leanprover-community/mathlib4", 
    "609f87d46c764e488d0034a754b2d3989c9e883f"
)

print("Tracing Mathlib4... This uses remote cache if available.")
traced_repo = trace(repo)

data_by_theorem = {}
samples_per_domain = {domain: 0 for domain in TARGET_DOMAINS}
MAX_SAMPLES_PER_DOMAIN = 1000

for file in traced_repo.traced_files:
    file_path = str(file.path)
    current_domain = next((d for d, path in TARGET_DOMAINS.items() if path in file_path), None)
    if not current_domain or samples_per_domain[current_domain] >= MAX_SAMPLES_PER_DOMAIN:
        continue

    print(f"[{current_domain}] Processing: {file.path}")
    theorems = file.get_traced_theorems()
    
    for thm in theorems:
        # thm.name instead of thm.theorem_name
        thm_name = getattr(thm, "name", "Unknown")
        tactics = thm.get_traced_tactics()

        pairs = []
        for tactic in tactics:
            state = tactic.state_before
            if state:
                y = extract_variable_count(state)
                pairs.append({"state": state, "label": y})
            
        if pairs:
            data_by_theorem[thm_name] = pairs
            
            # We only need one to prove the pipeline works!

all_theorems = list(data_by_theorem.keys())
random.seed(42)
random.shuffle(all_theorems)

n = len(all_theorems)
train_end = int(n * 0.7)
val_end = int(n * 0.85)

train_thms = all_theorems[:train_end]
val_thms = all_theorems[train_end:val_end]
test_thms = all_theorems[val_end:]

def flatten_data(theorem_list):
    return [pair for thm in theorem_list for pair in data_by_theorem[thm]]

dataset = {
    "train": flatten_data(train_thms),
    "val": flatten_data(val_thms),
    "test": flatten_data(test_thms)
}

output_path = "gold_dataset.json"
with open(output_path, "w") as f:
    json.dump(dataset, f, indent=2)

print(f"\nHarvest Complete!")
print(f"Total Theorems: {n}")
print(f"Train samples: {len(dataset['train'])}")
print(f"Val samples:   {len(dataset['val'])}")
print(f"Test samples:  {len(dataset['test'])}")