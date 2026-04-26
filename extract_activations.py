import torch
from transformer_lens import HookedTransformer

def get_activations(model, prompt, layer=18):
    """
    Optimized extraction: Only caches the required layer and 
    extracts the last token's residual stream.
    """
    # Use the model's tokenizer
    tokens = model.to_tokens(prompt) 
    
    # names_filter ensures we ONLY store the layer we care about
    # which prevents RAM from filling up
    with torch.no_grad():
        _, cache = model.run_with_cache(
            tokens, 
            names_filter=lambda name: name == f"blocks.{layer}.hook_resid_post"
        )

    # Get the residual stream for the specified layer
    # Shape: [batch, position, d_model]
    resid_post = cache[f"blocks.{layer}.hook_resid_post"]
    
    # Extract the LAST token's activation (index -1)
    # This is usually where the "summary" of the proof state lives
    last_token_act = resid_post[0, -1, :] 
    
    return last_token_act

def prepare_dataset(model, prompts, labels, layer=18):
    """
    Processes a list of Lean prompts into a tensor of activations.
    """
    activations = []
    print(f"Extracting layer {layer} activations...")
    
    for i, p in enumerate(prompts):
        act = get_activations(model, p, layer)
        activations.append(act)
        if (i + 1) % 5 == 0:
            print(f"  Processed {i + 1}/{len(prompts)} prompts...")

    X = torch.stack(activations)
    y = torch.tensor(labels).float().view(-1, 1)
    return X, y