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

repo = LeanGitRepo(
    "https://github.com/yangky11/lean4-example", 
    "7b6ecb9ad4829e4e73600a3329baeb3b5df8d23f"
)

print("Tracing repository...")
traced_repo = trace(repo)

data_by_theorem = {}

for file in traced_repo.traced_files:
    if ".lake/packages" in str(file.path):
        continue

    print(f"Processing: {file.path}")
    theorems = file.get_traced_theorems()
    
    for thm in theorems:
        # thm.name instead of thm.theorem_name
        name = getattr(thm, "name", "Unknown Theorem")
        
        # Method call instead of attribute
        tactics = thm.get_traced_tactics()
        
        for tactic in tactics:
            print("-" * 30)
            print(f"Theorem: {name}")
            print(f"Tactic: {tactic.tactic}")
            print(f"State Before: \n{tactic.state_before}")
            
            # We only need one to prove the pipeline works!
            exit()