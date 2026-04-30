import torch
from transformer_lens import HookedTransformer
from extract_activations import prepare_dataset
import torch.nn as nn
import torch.optim as optim
import json
import os

def train_general_probe(X, y, output_dim=1, task='regression'):
    probe = nn.Linear(X.shape[1], output_dim)
    optimizer = optim.Adam(probe.parameters(), lr=0.001)
    if task == 'regression':
        criterion = nn.MSELoss()
    elif task == 'binary':
        criterion = nn.BCEWithLogitsLoss()
    else:
        raise ValueError("Task must be 'regression' or 'binary'")

    print(f"Training {task} probe on {X.shape[0]} samples...")
    for epoch in range(101):
        optimizer.zero_grad()
        predictions = probe(X)
        loss = criterion(predictions, y)
        loss.backward()
        optimizer.step()
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
    return probe

def evaluate_probe(probe, X, y, task='regression'):
    with torch.no_grad():
        preds = probe(X)
        if task == 'regression':
            mse = torch.mean((preds - y)**2).item()
            return mse
        elif task == 'binary':
            acc = ((torch.sigmoid(preds) > 0.5) == y).float().mean().item()
            return acc

def get_shuffled_labels(labels):
    shuffled = labels.clone()
    torch.manual_seed(42)
    shuffled = shuffled[torch.randperm(shuffled.size(0))]
    return shuffled

def train_token_probe(X, y_var, y_kwd, input_dim=768):
    probe_var = nn.Linear(input_dim, 1)
    probe_kwd = nn.Linear(input_dim, 1)
    optimizer = optim.Adam(list(probe_var.parameters()) + list(probe_kwd.parameters()), lr=0.001)
    criterion = nn.BCEWithLogitsLoss()
    
    print(f"Training token-level probe on {X.shape[0]} tokens...")
    for epoch in range(101):
        optimizer.zero_grad()
        pred_var = probe_var(X)
        pred_kwd = probe_kwd(X)
        loss = criterion(pred_var, y_var) + criterion(pred_kwd, y_kwd)
        loss.backward()
        optimizer.step()
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
    return probe_var, probe_kwd

def load_prompts(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)

    train_prompts = [item['state'] for item in data.get('train',[])]
    train_labels = [item.get('labels', {'variable_count': item.get('label', 0), 'is_var': [], 'is_kwd': []}) for item in data.get('train', [])]

    test_prompts = [item['state'] for item in data.get('test',[])]
    test_labels = [item.get('labels', {'variable_count': item.get('label', 0), 'is_var': [], 'is_kwd': []}) for item in data.get('test',[])]

    return train_prompts, train_labels, test_prompts, test_labels

def main():
    device = "cpu"
    model_name = "Qwen/Qwen2.5-0.5B"
    
    print(f"Loading {model_name}...")
    model = HookedTransformer.from_pretrained(
        model_name,
        device=device,
        fold_ln=False,
        center_writing_weights=False,
        hf_model=None,
        torch_dtype=torch.bfloat16,
    )

    train_prompts, train_labels, test_prompts, test_labels = load_prompts("gold_dataset.json")
    cache_file = "activations_cache_layer18.pt"
    rebuild_cache = False
    if os.path.exists(cache_file):
        data = torch.load(cache_file)
        if data['X_train'].shape[0] != len(train_prompts) or data['X_test'].shape[0] != len(test_prompts):
            print("Cache size mismatch detected. Recomputing activations for the current dataset.")
            rebuild_cache = True
        else:
            X_train, y_train = data['X_train'], data['y_train']
            X_test, y_test = data['X_test'], data['y_test']
    else:
        rebuild_cache = True

    if rebuild_cache:
        print("Cache not found or stale. Running model extraction (this may take a while)...")
        X_train, y_train = prepare_dataset(model, train_prompts, train_labels, layer=18)
        X_test, y_test = prepare_dataset(model, test_prompts, test_labels, layer=18)
        X_train, y_train = X_train.float(), y_train.float()
        X_test, y_test = X_test.float(), y_test.float()

        torch.save({
            'X_train': X_train,
            'y_train': y_train,
            'X_test': X_test,
            'y_test': y_test,
        }, cache_file)
        print(f"Extraction complete. Data saved to {cache_file}")

    X_mean = X_train.mean(dim=0)
    X_std = X_train.std(dim=0) + 1e-6
    X_train_norm = (X_train - X_mean) / X_std
    X_test_norm = (X_test - X_mean) / X_std

    print("\n--- BASELINE VARIABLE COUNT PROBE ---")
    probe = train_general_probe(X_train_norm, y_train, output_dim=1, task='regression')
    mse = evaluate_probe(probe, X_test_norm, y_test, task='regression')
    print(f"Layer 18 Variable Count MSE: {mse:.4f}")

    # Additional feature probes and shuffled controls
    feature_names = [
        'goal_depth',
        'quantifier_presence',
        'context_size',
        'type_complexity',
        'inductive_vs_direct',
        'equality_vs_inequality',
        'unbound_variables',
    ]

    if all(isinstance(l, dict) and all(feat in l for feat in feature_names) for l in train_labels):
        print("\n--- FEATURE SWEEP AND SHUFFLED CONTROLS ---")
        for feat in feature_names:
            y_train_feat = torch.tensor([float(l[feat]) for l in train_labels]).view(-1, 1)
            y_test_feat = torch.tensor([float(l[feat]) for l in test_labels]).view(-1, 1)
            y_mean = y_train_feat.mean()
            y_std = y_train_feat.std() + 1e-6
            y_train_norm = (y_train_feat - y_mean) / y_std
            y_test_norm = (y_test_feat - y_mean) / y_std

            print(f"\nProbing {feat}...")
            probe_feat = train_general_probe(X_train_norm, y_train_norm, output_dim=1, task='regression')
            mse_feat = evaluate_probe(probe_feat, X_test_norm, y_test_norm, task='regression')
            print(f"{feat} MSE: {mse_feat:.4f}")

            y_shuffled = get_shuffled_labels(y_train_norm)
            probe_shuffled = train_general_probe(X_train_norm, y_shuffled, output_dim=1, task='regression')
            mse_shuffled = evaluate_probe(probe_shuffled, X_test_norm, y_test_norm, task='regression')
            print(f"{feat} Shuffled MSE: {mse_shuffled:.4f}")
    else:
        print("\nSkipping feature sweep: dataset does not contain structured labels.")

    # Syntactic control feature: character count
    char_count_train = torch.tensor([len(p) for p in train_prompts]).float().view(-1, 1)
    char_count_test = torch.tensor([len(p) for p in test_prompts]).float().view(-1, 1)
    char_mean = char_count_train.mean()
    char_std = char_count_train.std() + 1e-6
    char_train_norm = (char_count_train - char_mean) / char_std
    char_test_norm = (char_count_test - char_mean) / char_std

    print("\n--- SYNTACTIC CONTROL ---")
    probe_char = train_general_probe(X_train_norm, char_train_norm, output_dim=1, task='regression')
    mse_char = evaluate_probe(probe_char, X_test_norm, char_test_norm, task='regression')
    print(f"Character Count MSE: {mse_char:.4f}")

    # Layer 0 baseline
    cache_file_l0 = "activations_cache_layer0.pt"
    rebuild_l0 = False
    if os.path.exists(cache_file_l0):
        data = torch.load(cache_file_l0)
        if data['X_train_l0'].shape[0] != len(train_prompts) or data['X_test_l0'].shape[0] != len(test_prompts):
            print("Layer 0 cache size mismatch detected. Recomputing activations.")
            rebuild_l0 = True
        else:
            X_train_l0 = data['X_train_l0']
            X_test_l0 = data['X_test_l0']
    else:
        rebuild_l0 = True

    if rebuild_l0:
        print("Layer 0 cache not found or stale. Extracting...")
        X_train_l0, y_train_l0 = prepare_dataset(model, train_prompts, train_labels, layer=0)
        X_test_l0, y_test_l0 = prepare_dataset(model, test_prompts, test_labels, layer=0)
        X_train_l0, y_train_l0 = X_train_l0.float(), y_train_l0.float()
        X_test_l0, y_test_l0 = X_test_l0.float(), y_test_l0.float()
        torch.save({'X_train_l0': X_train_l0, 'X_test_l0': X_test_l0}, cache_file_l0)

    X_l0_mean = X_train_l0.mean(dim=0)
    X_l0_std = X_train_l0.std(dim=0) + 1e-6
    X_train_l0_norm = (X_train_l0 - X_l0_mean) / X_l0_std
    X_test_l0_norm = (X_test_l0 - X_l0_mean) / X_l0_std

    print("\n--- LAYER 0 BASELINE ---")
    probe_l0 = train_general_probe(X_train_l0_norm, y_train, output_dim=1, task='regression')
    mse_l0 = evaluate_probe(probe_l0, X_test_l0_norm, y_test, task='regression')
    print(f"Layer 0 MSE: {mse_l0:.4f}")

    # Token-level probe
    if any(l.get('is_var') for l in train_labels):
        cache_file_token = "activations_cache_token_layer18.pt"
        rebuild_token = False
        expected_train_tokens = sum(len(l['is_var']) for l in train_labels)
        expected_test_tokens = sum(len(l['is_var']) for l in test_labels)

        if os.path.exists(cache_file_token):
            data = torch.load(cache_file_token)
            if data['X_train_token'].shape[0] != expected_train_tokens or data['X_test_token'].shape[0] != expected_test_tokens:
                print("Token cache size mismatch detected. Recomputing token activations.")
                rebuild_token = True
            else:
                X_train_token = data['X_train_token']
                y_var_train = data['y_var_train']
                y_kwd_train = data['y_kwd_train']
                X_test_token = data['X_test_token']
                y_var_test = data['y_var_test']
                y_kwd_test = data['y_kwd_test']
        else:
            rebuild_token = True

        if rebuild_token:
            print("Token cache not found or stale. Running token extraction...")
            X_train_token, y_var_train, y_kwd_train = prepare_dataset(model, train_prompts, train_labels, layer=18, all_positions=True)
            X_test_token, y_var_test, y_kwd_test = prepare_dataset(model, test_prompts, test_labels, layer=18, all_positions=True)
            torch.save({
                'X_train_token': X_train_token,
                'y_var_train': y_var_train,
                'y_kwd_train': y_kwd_train,
                'X_test_token': X_test_token,
                'y_var_test': y_var_test,
                'y_kwd_test': y_kwd_test,
            }, cache_file_token)
            print(f"Token extraction complete. Data saved to {cache_file_token}")

        X_train_token_norm = (X_train_token - X_train_token.mean(dim=0)) / (X_train_token.std(dim=0) + 1e-6)
        X_test_token_norm = (X_test_token - X_train_token.mean(dim=0)) / (X_train_token.std(dim=0) + 1e-6)
        probe_var, probe_kwd = train_token_probe(X_train_token_norm, y_var_train, y_kwd_train, input_dim=X_train_token.shape[1])

        with torch.no_grad():
            pred_var = torch.sigmoid(probe_var(X_test_token_norm))
            pred_kwd = torch.sigmoid(probe_kwd(X_test_token_norm))
            acc_var = ((pred_var > 0.5) == y_var_test).float().mean().item()
            acc_kwd = ((pred_kwd > 0.5) == y_kwd_test).float().mean().item()
            print("\n--- TOKEN-LEVEL RESULTS ---")
            print(f"Variable Accuracy: {acc_var:.4f}")
            print(f"Keyword Accuracy: {acc_kwd:.4f}")
    else:
        print("Skipping token-level probe: no token labels in dataset.")

if __name__ == "__main__":
    main()