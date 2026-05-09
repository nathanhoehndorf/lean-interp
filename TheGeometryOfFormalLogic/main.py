import json
import re
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from config import ProbeConfig
from extract_activations import prepare_dataset
from features import FEATURES
from loader import load_prompts
from model import load_model

import pandas as pd
import seaborn as sns

def test_feature_correlations(labels: list[dict], output_dir: Path):
    df = pd.DataFrame(labels)
    corr_matrix = df.corr()

    print("\n--- Feature Correlation Matrix ---")
    print(corr_matrix)

    plt.figure(figsize=(10,8))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Feature Correlations")
    plt.savefig(output_dir / "feature_correlation_heatmap.png")
    plt.show()

    return corr_matrix

def train_general_probe(X: torch.Tensor, y: torch.Tensor, output_dim: int = 1, task: str = "regression") -> nn.Module:
    probe = nn.Linear(X.shape[1], output_dim)
    optimizer = optim.Adam(probe.parameters(), lr=0.001)
    criterion = nn.MSELoss() if task == "regression" else nn.BCEWithLogitsLoss()

    print(f"Training {task} probe on {X.shape[0]} samples...")
    for epoch in range(101):
        optimizer.zero_grad()
        predictions = probe(X)
        loss = criterion(predictions, y)
        loss.backward()
        optimizer.step()
        if epoch % 20 == 0:
            print(f"  Epoch {epoch}, Loss: {loss.item():.4f}")
    return probe


def evaluate_probe(probe: nn.Module, X: torch.Tensor, y: torch.Tensor, task: str = "regression") -> float:
    with torch.no_grad():
        predictions = probe(X)
        if task == "regression":
            return torch.mean((predictions - y) ** 2).item()
        return ((torch.sigmoid(predictions) > 0.5) == y).float().mean().item()


def train_logistic_regression(X: torch.Tensor, y: torch.Tensor, epochs: int = 101, lr: float = 1e-3) -> nn.Module:
    probe = nn.Linear(X.shape[1], 1)
    optimizer = optim.Adam(probe.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    print(f"Training logistic regression on {X.shape[0]} samples...")
    for epoch in range(epochs):
        optimizer.zero_grad()
        logits = probe(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        if epoch % 20 == 0:
            print(f"  Epoch {epoch}, Loss: {loss.item():.4f}")

    return probe


def evaluate_logistic_regression(probe: nn.Module, X: torch.Tensor, y: torch.Tensor) -> float:
    with torch.no_grad():
        logits = probe(X)
        predictions = torch.sigmoid(logits)
        return ((predictions > 0.5) == y).float().mean().item()


def normalize_dataset(X_train: torch.Tensor, X_test: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mean = X_train.mean(dim=0, keepdim=True)
    std = X_train.std(dim=0, keepdim=True) + 1e-6
    return (X_train - mean) / std, (X_test - mean) / std


def sanitize_model_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", name)


def make_result_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(data: dict, output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def plot_binary_heatmap(matrix: torch.Tensor, layer_labels: list[int], feature_labels: list[str], out_path: Path, title: str) -> None:
    plt.figure(figsize=(max(8, len(layer_labels) * 0.35), max(4, len(feature_labels) * 0.7)))
    plt.imshow(matrix, cmap="viridis", aspect="auto", vmin=0.0, vmax=1.0)
    plt.colorbar(label="Accuracy")
    plt.xticks(range(len(layer_labels)), layer_labels, rotation=45)
    plt.yticks(range(len(feature_labels)), feature_labels)
    plt.title(title)
    plt.xlabel("Layer")
    plt.ylabel("Binary Feature")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_regression_lines(metrics: dict[str, list[float]], layer_labels: list[int], baseline: list[float] | None, out_path: Path, title: str) -> None:
    plt.figure(figsize=(10, 6))
    for name, values in metrics.items():
        plt.plot(layer_labels, values, marker="o", label=name)
    if baseline is not None:
        plt.plot(layer_labels, baseline, marker="x", linestyle="--", label="Character count baseline")
    plt.title(title)
    plt.xlabel("Layer")
    plt.ylabel("Normalized MSE")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def compute_pca_3d(X: torch.Tensor) -> torch.Tensor:
    X = X - X.mean(dim=0, keepdim=True)
    u, s, vh = torch.linalg.svd(X, full_matrices=False)
    # torch.linalg.svd returns Vh (right singular vectors transposed).
    # Project onto the first three principal components.
    return X @ vh[:3, :].T


def plot_pca_activations(X_train: torch.Tensor, X_test: torch.Tensor, layer: int, out_path: Path, title: str, sample_size: int = 500) -> None:
    def sample_points(X: torch.Tensor, limit: int) -> torch.Tensor:
        if X.shape[0] <= limit:
            return X
        indices = torch.randperm(X.shape[0], device=X.device)[:limit]
        return X[indices]

    train_samples = sample_points(X_train, sample_size)
    test_samples = sample_points(X_test, sample_size)
    combined = torch.cat([train_samples, test_samples], dim=0)
    projected = compute_pca_3d(combined).cpu().numpy()

    split = train_samples.shape[0]
    elev, azim = 30, 30

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    ax.scatter(projected[:split, 0], projected[:split, 1], projected[:split, 2], s=8, alpha=0.6, label="train", c="#1f77b4")
    ax.scatter(projected[split:, 0], projected[split:, 1], projected[split:, 2], s=8, alpha=0.6, label="test", c="#ff7f0e")
    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.view_init(elev=elev, azim=azim)
    ax.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def get_layer_indices(config: ProbeConfig, model) -> list[int]:
    if not config.sweep_layers:
        return [config.layer]

    if hasattr(model, "cfg") and getattr(model.cfg, "n_layers", None) is not None:
        n_layers = model.cfg.n_layers
    elif hasattr(model, "config") and getattr(model.config, "is_encoder_decoder", False):
        n_layers = getattr(model.config, "encoder_layers", None) or getattr(model.config, "num_layers", None)
        if n_layers is None and hasattr(model, "encoder") and hasattr(model.encoder, "block"):
            n_layers = len(model.encoder.block)
    else:
        n_layers = len(getattr(model, "blocks", []))

    if n_layers is None:
        raise ValueError("Unable to infer number of layers from the loaded model.")

    max_layer = config.max_layer if config.max_layer is not None else n_layers - 1
    max_layer = min(max_layer, n_layers - 1)
    return list(range(0, max_layer + 1))


def build_cache_filename(config: ProbeConfig, layers: list[int]) -> Path:
    model_name_safe = sanitize_model_name(config.model_name)
    layer_suffix = str(layers[0]) if len(layers) == 1 else f"{layers[0]}_{layers[-1]}"
    return Path(f"data/activations_cache_{config.mode.value}_{model_name_safe}_layers{layer_suffix}.pt")


def maybe_load_cache(cache_path: Path, expected_train: int, expected_test: int, num_layers: int):
    if not cache_path.exists():
        return None
    data = torch.load(cache_path)
    if (
        data.get("layers") is None
        or data["layers"] != num_layers
        or data["X_train"].shape[0] != expected_train
        or data["X_test"].shape[0] != expected_test
    ):
        return None
    return data["X_train"], data["y_train"], data["X_test"], data["y_test"]


def load_or_build_dataset(cache_path: Path, model, train_prompts: list[str], train_labels: list[dict], test_prompts: list[str], test_labels: list[dict], layers: list[int]):
    cached = maybe_load_cache(cache_path, len(train_prompts), len(test_prompts), len(layers))
    if cached is not None:
        return cached

    print("Cache not found or stale. Running model extraction (this may take a while)...")
    X_train, y_train = prepare_dataset(model, train_prompts, train_labels, layers=layers)
    X_test, y_test = prepare_dataset(model, test_prompts, test_labels, layers=layers)
    X_train, y_train = X_train.float(), y_train.float()
    X_test, y_test = X_test.float(), y_test.float()
    torch.save({
        "layers": len(layers),
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
    }, cache_path)
    print(f"Extraction complete. Data saved to {cache_path}")
    return X_train, y_train, X_test, y_test


def run_layer_probe(model, train_prompts, train_labels, test_prompts, test_labels, layers: list[int], output_dir: Path):
    cache_file = build_cache_filename(config, layers)
    X_train, y_train, X_test, y_test = load_or_build_dataset(cache_file, model, train_prompts, train_labels, test_prompts, test_labels, layers)
    X_train_norm, X_test_norm = normalize_dataset(X_train, X_test)

    binary_features = [feature for feature in FEATURES if feature.task == "binary"]
    regression_features = [feature for feature in FEATURES if feature.task == "regression"]

    binary_matrix = torch.zeros((len(binary_features), len(layers)), dtype=torch.float32)
    binary_logistic_matrix = torch.zeros((len(binary_features), len(layers)), dtype=torch.float32)
    regression_metrics: dict[str, list[float]] = {}
    char_count_baseline: list[float] = []

    print("\n--- LAYER PROBE SWEEP ---")
    for layer_index, layer in enumerate(layers):
        X_train_layer = X_train_norm[:, layer_index, :]
        X_test_layer = X_test_norm[:, layer_index, :]

        char_count_train = torch.tensor([len(p) for p in train_prompts]).float().view(-1, 1)
        char_count_test = torch.tensor([len(p) for p in test_prompts]).float().view(-1, 1)
        char_mean = char_count_train.mean()
        char_std = char_count_train.std() + 1e-6
        char_train_norm = (char_count_train - char_mean) / char_std
        char_test_norm = (char_count_test - char_mean) / char_std

        probe_char = train_general_probe(X_train_layer, char_train_norm, output_dim=1, task="regression")
        char_count_baseline.append(evaluate_probe(probe_char, X_test_layer, char_test_norm, task="regression"))

        for feature_index, feature in enumerate(binary_features):
            y_train_feat = torch.tensor([feature.label_value(train_labels[k], train_prompts[k]) for k in range(len(train_prompts))]).float().view(-1, 1)
            y_test_feat = torch.tensor([feature.label_value(test_labels[k], test_prompts[k]) for k in range(len(test_prompts))]).float().view(-1, 1)

            probe = train_general_probe(X_train_layer, y_train_feat, output_dim=1, task="binary")
            acc = evaluate_probe(probe, X_test_layer, y_test_feat, task="binary")
            binary_matrix[feature_index, layer_index] = acc
            print(f"Layer {layer} {feature.name} accuracy: {acc:.4f}")

            logistic_probe = train_logistic_regression(X_train_layer, y_train_feat)
            logistic_acc = evaluate_logistic_regression(logistic_probe, X_test_layer, y_test_feat)
            binary_logistic_matrix[feature_index, layer_index] = logistic_acc
            print(f"Layer {layer} {feature.name} logistic regression accuracy: {logistic_acc:.4f}")

        for feature in regression_features:
            y_train_feat = torch.tensor([feature.label_value(train_labels[k], train_prompts[k]) for k in range(len(train_prompts))]).float().view(-1, 1)
            y_test_feat = torch.tensor([feature.label_value(test_labels[k], test_prompts[k]) for k in range(len(test_prompts))]).float().view(-1, 1)
            y_mean = y_train_feat.mean()
            y_std = y_train_feat.std() + 1e-6
            y_train_norm = (y_train_feat - y_mean) / y_std
            y_test_norm = (y_test_feat - y_mean) / y_std
            probe = train_general_probe(X_train_layer, y_train_norm, output_dim=1, task="regression")
            mse = evaluate_probe(probe, X_test_layer, y_test_norm, task="regression")
            regression_metrics.setdefault(feature.name, []).append(mse)
            print(f"Layer {layer} {feature.name} MSE: {mse:.4f}")

        pca_path = output_dir / f"pca_activations_layer{layer}_{config.mode.value}.png"
        plot_pca_activations(
            X_train_norm[:, layer_index, :],
            X_test_norm[:, layer_index, :],
            layer,
            pca_path,
            f"PCA of activation space for layer {layer} ({config.mode.value})",
        )

    if binary_features:
        heatmap_path = output_dir / f"binary_accuracy_heatmap_{config.mode.value}.png"
        plot_binary_heatmap(binary_matrix, layers, [feature.name for feature in binary_features], heatmap_path, f"Binary Feature Accuracy ({config.mode.value})")

        logistic_heatmap_path = output_dir / f"binary_logistic_accuracy_heatmap_{config.mode.value}.png"
        plot_binary_heatmap(binary_logistic_matrix, layers, [feature.name for feature in binary_features], logistic_heatmap_path, f"Logistic Regression Accuracy ({config.mode.value})")

    if regression_features:
        regression_path = output_dir / f"regression_mse_{config.mode.value}.png"
        plot_regression_lines(regression_metrics, layers, char_count_baseline, regression_path, f"Regression Probe MSE ({config.mode.value})")

    return {
        "mode": config.mode.value,
        "layers": layers,
        "binary": {feature.name: binary_matrix[i].tolist() for i, feature in enumerate(binary_features)},
        "binary_logistic": {feature.name: binary_logistic_matrix[i].tolist() for i, feature in enumerate(binary_features)},
        "regression": regression_metrics,
        "char_count_baseline": char_count_baseline,
    }


def main() -> None:
    global config
    config = ProbeConfig.from_args()
    print(f"Using config: {config}")

    print(f"Loading {config.model_name} on {config.device}...")
    model = load_model(config)

    train_prompts, train_labels, test_prompts, test_labels = load_prompts(config.dataset_path, sample_size=config.sample_size)
    if not train_prompts or not test_prompts:
        raise ValueError("The dataset is empty or the sample size is too small.")

    layers = get_layer_indices(config, model)
    result_dir = Path(config.output_dir)
    make_result_dir(result_dir)

    test_feature_correlations(train_labels, result_dir)

    results = run_layer_probe(model, train_prompts, train_labels, test_prompts, test_labels, layers, result_dir)
    results_path = result_dir / f"probe_results_{config.mode.value}.json"
    save_json(results, results_path)
    print(f"Results written to {results_path}")

    print("\nCompleted probe run. Generated layer-wise figures and metrics.")


if __name__ == "__main__":
    main()
