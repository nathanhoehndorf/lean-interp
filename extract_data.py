import os
from lean_dojo_v2.lean_dojo import LeanGitRepo, trace

repo = LeanGitRepo(
    "https://github.com/yangky11/lean4-example", 
    "7b6ecb9ad4829e4e73600a3329baeb3b5df8d23f"
)

print("Tracing repository...")
traced_repo = trace(repo)

for file in traced_repo.traced_files:
    file_path_str = str(file.path)
    if ".lake/packages" in file_path_str:
        continue

    print(f"\nChecking file: {file_path_str}")
    
    # Get the theorems in this file
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