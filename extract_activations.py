import torch
from transformer_lens import HookedTransformer

def get_activations(model, prompt, layer=6):
    """
    Runs the model and extracts the residual stream for a specific layer.
    """
    tokens = model.to_tokens(prompt)
    _, cache = model.run_with_cache(prompt)

    resid_post = cache["resid_post", layer]
    mean_act = resid_post[0].mean(dim=0)
    return mean_act

def prepare_dataset(model, prompts, labels, layer=6):
    """
    Processes a list of Lean prompts into a tensor of activations.
    """
    activations = []
    for p in prompts:
        act = get_activations(model, p, layer)
        activations.append(act)

    X = torch.stack(activations)
    y = torch.tensor(labels).float().view(-1,1)
    return X,y