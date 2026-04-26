import torch
from transformer_lens import HookedTransformer
from extract_activations import prepare_dataset
import torch.nn as nn
import torch.optim as optim

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

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "Qwen/Qwen2.5-1.5B"
    
    print(f"Loading {model_name}...")
    model = HookedTransformer.from_pretrained(model_name, device=device, fold_ln=False, center_writing_weights=False, hf_model=None, torch_dtype=torch.bfloat16)

    # Example: Proof states with different variable counts
    prompts = [
        "x : Nat ⊢ x = x",
        "a b : Nat ⊢ a = b",
        "var1 var2 var3 : Nat ⊢ var1 = var3",
        "n m o p : Nat ⊢ n + m = o + p",
        "a b c d e : Nat ⊢ a = e", # 5 vars
        "x y z : Nat ⊢ x = z"
    ]
    labels = [1.0, 2.0, 3.0, 4.0, 5.0, 3.0]

    # 1. Extract Activations
    X, y = prepare_dataset(model, prompts, labels, layer=18)

    X_mean = X.mean(dim=0)
    X_std = X.std(dim=0) + 1e-6
    X_norm = (X - X_mean) / X_std

    # 2. Train and Test the Probe
    probe = train_probe(X_norm, y, input_dim=model.cfg.d_model)

    # 3. Simple Validation
    test_prompt = "p q r s t u v : Nat ⊢ p = v" # 7 variables
    test_act = prepare_dataset(model, [test_prompt], [7], layer=18)[0]
    test_act_norm = (test_act - X_mean) / X_std
    prediction = probe(test_act_norm).item()
    
    print(f"\nTest Prompt: {test_prompt}")
    print(f"Predicted Var Count: {prediction:.2f} (Target: 7.0)")

if __name__ == "__main__":
    main()