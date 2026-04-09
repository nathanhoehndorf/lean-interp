from transformer_lens import HookedTransformer
import torch

# Using a smaller Pythia model for the 'smoke test'
# It's in your 'Valid official model names' list!
model_name = "EleutherAI/pythia-160m"

print(f"Loading {model_name}...")
model = HookedTransformer.from_pretrained(model_name, device="cpu") 

# This is your string from the previous step
prompt = "a b c : Nat ⊢ a + b + c = a + c + b"

# Run with cache to grab the 'Internal Brain States'
logits, cache = model.run_with_cache(prompt)

# Let's look at the Residual Stream at the middle layer
# For Pythia-160m, there are 12 layers. Let's grab Layer 6.
resid_post = cache["resid_post", 6]

print(f"Success! Activation Shape: {resid_post.shape}")
print("Sequence Length:", resid_post.shape[1])
print("Hidden Dimension (d_model):", resid_post.shape[2])