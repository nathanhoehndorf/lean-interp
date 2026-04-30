import torch
from transformer_lens import HookedTransformer

def get_activations(model, prompt, layer=18, all_positions=False):
    """
    Optimized extraction: Extracts activations from intermediate layers.
    If all_positions=True, returns all token positions; else, only the last token.
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
    
    if all_positions:
        # Return all token activations
        return resid_post[0]  # [seq_len, d_model]
    else:
        # Extract the LAST token's activation (index -1)
        # This is usually where the "summary" of the proof state lives
        last_token_act = resid_post[0, -1, :] 
        return last_token_act

def prepare_dataset(model, prompts, labels, layer=18, all_positions=False):
    """
    Processes a list of Lean prompts into a tensor of activations.
    If all_positions=True, returns flattened activations for all tokens across all prompts.
    """
    if all_positions:
        all_activations = []
        all_labels_var = []
        all_labels_kwd = []
        print(f"Extracting layer {layer} activations for all positions...")
        
        for i, p in enumerate(prompts):
            act = get_activations(model, p, layer, all_positions=True)  # [seq_len, d_model]
            var_labels = torch.tensor(labels[i]['is_var'], dtype=torch.float).unsqueeze(1)
            kwd_labels = torch.tensor(labels[i]['is_kwd'], dtype=torch.float).unsqueeze(1)

            if var_labels.size(0) != act.size(0) or kwd_labels.size(0) != act.size(0):
                raise ValueError(
                    f"Token-label length mismatch for prompt {i}: "
                    f"{var_labels.size(0)} var labels, {kwd_labels.size(0)} kwd labels, "
                    f"{act.size(0)} tokens."
                )

            all_activations.append(act)
            all_labels_var.append(var_labels)
            all_labels_kwd.append(kwd_labels)
            if (i + 1) % 5 == 0:
                print(f"  Processed {i + 1}/{len(prompts)} prompts...")
        
        X = torch.cat(all_activations, dim=0).float()  # [total_tokens, d_model]
        y_var = torch.cat(all_labels_var, dim=0).float()  # [total_tokens, 1]
        y_kwd = torch.cat(all_labels_kwd, dim=0).float()  # [total_tokens, 1]
        return X, y_var, y_kwd
    else:
        activations = []
        y = []
        print(f"Extracting layer {layer} activations for last token...")
        
        for i, p in enumerate(prompts):
            act = get_activations(model, p, layer, all_positions=False)
            activations.append(act)
            # For global, labels is scalar or dict, but for now assume scalar
            if isinstance(labels[i], dict):
                y.append(labels[i]['variable_count'])
            else:
                y.append(labels[i])
            if (i + 1) % 5 == 0:
                print(f"  Processed {i + 1}/{len(prompts)} prompts...")

        X = torch.stack(activations)
        y = torch.tensor(y).float().view(-1, 1)
        return X, y