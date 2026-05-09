import torch
from transformer_lens import HookedTransformer


def is_seq2seq_model(model) -> bool:
    return hasattr(model, "config") and getattr(model.config, "is_encoder_decoder", False)


def get_model_device(model):
    if hasattr(model, "device"):
        return model.device
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def get_activations(model, prompt, layers=None, all_positions=False):
    """
    Optimized extraction: Extracts activations from intermediate layers.
    If all_positions=True, returns all token positions for a single layer;
    otherwise it returns the final-token activations for one or more layers.
    """
    if layers is None:
        layers = [18]
    if isinstance(layers, int):
        layers = [layers]

    if isinstance(model, HookedTransformer):
        tokens = model.to_tokens(prompt)
        with torch.no_grad():
            _, cache = model.run_with_cache(
                tokens,
                names_filter=lambda name: any(name == f"blocks.{layer}.hook_resid_post" for layer in layers),
            )

        if all_positions:
            if len(layers) != 1:
                raise ValueError("all_positions=True supports only a single layer index.")
            resid_post = cache[f"blocks.{layers[0]}.hook_resid_post"]
            return resid_post[0]

        output = []
        for layer in layers:
            resid_post = cache[f"blocks.{layer}.hook_resid_post"]
            output.append(resid_post[0, -1, :])
        return torch.stack(output, dim=0)

    if not is_seq2seq_model(model):
        raise ValueError("Unsupported model type for activation extraction")

    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is None:
        raise ValueError("Seq2seq models must carry a tokenizer attribute")

    tokens = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True)
    device = get_model_device(model)
    tokens = {k: v.to(device) for k, v in tokens.items()}

    with torch.no_grad():
        encoder = model.get_encoder() if hasattr(model, "get_encoder") else model.encoder
        outputs = encoder(**tokens, output_hidden_states=True, return_dict=True)

    hidden_states = outputs.hidden_states
    if all_positions:
        if len(layers) != 1:
            raise ValueError("all_positions=True supports only a single layer index.")
        return hidden_states[layers[0] + 1][0]

    output = []
    for layer in layers:
        output.append(hidden_states[layer + 1][0, -1, :])
    return torch.stack(output, dim=0)


def prepare_dataset(model, prompts, labels, layers=None, all_positions=False, label_key='variable_count'):
    """
    Processes a list of Lean prompts into activations.
    If all_positions=True, returns flattened activations for all tokens across all prompts.
    """
    if layers is None:
        layers = [18]
    if isinstance(layers, int):
        layers = [layers]

    if all_positions:
        if len(layers) != 1:
            raise ValueError("all_positions=True supports only a single layer index.")

        all_activations = []
        all_labels_var = []
        all_labels_kwd = []
        print(f"Extracting layer {layers[0]} activations for all positions...")

        for i, p in enumerate(prompts):
            act = get_activations(model, p, layers=layers, all_positions=True)
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

        X = torch.cat(all_activations, dim=0).float()
        y_var = torch.cat(all_labels_var, dim=0).float()
        y_kwd = torch.cat(all_labels_kwd, dim=0).float()
        return X, y_var, y_kwd

    activations = []
    y = []
    print(f"Extracting layer {layers} activations for last token...")

    for i, p in enumerate(prompts):
        act = get_activations(model, p, layers=layers, all_positions=False)
        activations.append(act)
        if isinstance(labels[i], dict):
            y.append(labels[i].get(label_key, 0))
        else:
            y.append(labels[i])
        if (i + 1) % 5 == 0:
            print(f"  Processed {i + 1}/{len(prompts)} prompts...")

    X = torch.stack(activations)
    y = torch.tensor(y).float().view(-1, 1)
    return X, y
