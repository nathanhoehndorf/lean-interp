import torch
from transformer_lens import HookedTransformer
from extract_activations import prepare_dataset
import torch.nn as nn
import torch.optim as optim
import json
import os

def train_probe(X, y, input_dim=768):
    probe = nn.Linear(input_dim, 1)
    optimizer = optim.Adam(probe.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    print(f"Training probe on {X.shape[0]} samples...")
    for epoch in range(101):
        optimizer.zero_grad()
        predictions = probe(X)
        loss = criterion(predictions, y)
        loss.backward()
        optimizer.step()
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
    return probe

def load_prompts(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)

    train_prompts = [item['state'] for item in data.get('train',[])]
    train_labels = [item['label'] for item in data.get('train', [])]

    test_prompts = [item['state'] for item in data.get('test',[])]
    test_labels = [item['label'] for item in data.get('test',[])]

    return train_prompts[:500], train_labels[:500], test_prompts[:500], test_labels[:500]

def main():
    device = "cpu"
    model_name = "Qwen/Qwen2.5-0.5B"
    
    print(f"Loading {model_name}...")
    model = HookedTransformer.from_pretrained(model_name, device=device, fold_ln=False, center_writing_weights=False, hf_model=None, torch_dtype=torch.bfloat16)

    # Example: Proof states with different variable counts
    train_prompts, train_labels, test_prompts, test_labels = load_prompts("gold_dataset.json")

    cache_file = "activations_cache_layer18.pt"

    if not os.path.exists(cache_file):
        print("Cache not found. Running model extraction (this may take a while)...")
        # Extract activations for BOTH train and test sets
        X_train, y_train = prepare_dataset(model, train_prompts, train_labels, layer=18)
        X_test, y_test = prepare_dataset(model, test_prompts, test_labels, layer=18)

        X_train, y_train = X_train.float(), y_train.float()
        X_test, y_test = X_test.float(), y_test.float()
        
        # Save everything to disk so we never do this again
        torch.save({
            'X_train': X_train, 'y_train': y_train,
            'X_test': X_test, 'y_test': y_test
        }, cache_file)
        print(f"Extraction complete. Data saved to {cache_file}")
    else:
        print("Loading activations from cache...")
        data = torch.load(cache_file)
        X_train, y_train = data['X_train'], data['y_train']
        X_test, y_test = data['X_test'], data['y_test']

    X_mean = X_train.mean(dim=0)
    X_std = X_train.std(dim=0) + 1e-6
    X_train_norm = (X_train - X_mean) / X_std

    probe = train_probe(X_train_norm, y_train, input_dim=X_train.shape[1])

    with torch.no_grad():
        X_test_norm = (X_test - X_mean) / X_std
        preds = probe(X_test_norm)
        mse = torch.mean((preds - y_test)**2).item()

        print("\n--- RESULTS ---")
        print(f"Layer 18 | MSE: {mse:.4f}")

if __name__ == "__main__":
    main()