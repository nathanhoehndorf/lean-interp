import json
from collections import Counter

file_path = 'data/states_gold_dataset.json'

stats = {
    "quantifier_presence": Counter(),
    "equality_vs_inequality": Counter(),
    "inductive_structure": Counter()
}

with open(file_path, 'r') as f:
    data = json.load(f)
    for record in data.get('train', []):
        labels = record.get('labels', {})
        for feature in stats.keys():
            val = labels[feature]
            stats[feature][val] += 1

print(f"{'Feature':<25} | {'0.0 Count':<10} | {'1.0 Count':<10} | {'Baseline Acc':<12}")
print("-" * 65)

for feature, counts in stats.items():
    total = sum(counts.values())
    if total == 0: continue
    
    majority_class_count = max(counts.values())
    baseline_accuracy = majority_class_count / total
    
    print(f"{feature:<25} | {counts[0.0]:<10} | {counts[1.0]:<10} | {baseline_accuracy:.2%}")