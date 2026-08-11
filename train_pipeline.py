import argparse
import copy
import csv
import json
import os
import random
from pathlib import Path

# Required by CUDA for deterministic matrix operations when a GPU is available.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import joblib
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


CLASS_NAMES = ["DAB", "DVB-T", "FM", "GSM", "LTE", "TETRA"]
CLASS_TO_ID = {name: index for index, name in enumerate(CLASS_NAMES)}
FEATURE_COUNT = 33
LATENT_DIM = 16
DEFAULT_EPOCHS = 550
SMOKE_EPOCHS = 2
PATIENCE = 10
LEARNING_RATE = 0.001
DROPOUT_RATE = 0.001
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.1


class Autoencoder(nn.Module):
    """33-feature autoencoder with a 16-dimensional encoded representation."""

    def __init__(self, input_dim=FEATURE_COUNT, compressed_dim=LATENT_DIM):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, compressed_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(compressed_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
        )

    def encode(self, features):
        return self.encoder(features)

    def forward(self, features):
        return self.decoder(self.encode(features))


class LSTMClassifier(nn.Module):
    """Classifier for a 16-step sequence containing one value per step."""

    def __init__(self, num_classes=len(CLASS_NAMES)):
        super().__init__()
        self.lstm_32 = nn.LSTM(input_size=1, hidden_size=32, batch_first=True)
        self.lstm_16 = nn.LSTM(input_size=32, hidden_size=16, batch_first=True)
        self.dense_16 = nn.Linear(16, 16)
        self.dropout = nn.Dropout(DROPOUT_RATE)
        self.output = nn.Linear(16, num_classes)

    def forward(self, latent_sequence):
        sequence, _ = self.lstm_32(latent_sequence)
        sequence, _ = self.lstm_16(sequence)
        features = torch.relu(self.dense_16(sequence[:, -1, :]))
        return self.output(self.dropout(features))


def set_random_seed(seed):
    """Seed Python, NumPy, and PyTorch for repeatable experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:  # Support older PyTorch versions that lack warn_only.
        torch.use_deterministic_algorithms(True)


def select_device(requested="auto"):
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is unavailable")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _column_name(fieldnames, candidates):
    normalized = {
        "".join(character for character in field.lower() if character.isalnum()): field
        for field in fieldnames
    }
    for candidate in candidates:
        key = "".join(character for character in candidate.lower() if character.isalnum())
        if key in normalized:
            return normalized[key]
    return None


def _label_id(value):
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        index = int(value)
        if 0 <= index < len(CLASS_NAMES):
            return index

    text = str(value).strip()
    compact = "".join(character for character in text.lower() if character.isalnum())
    aliases = {
        "dab": "DAB",
        "dvbt": "DVB-T",
        "fm": "FM",
        "gsm": "GSM",
        "lte": "LTE",
        "tetra": "TETRA",
    }
    if compact in aliases:
        return CLASS_TO_ID[aliases[compact]]

    try:
        numeric = float(text)
    except ValueError:
        numeric = None
    if numeric is not None and numeric.is_integer() and 0 <= numeric < len(CLASS_NAMES):
        return int(numeric)

    expected = ", ".join(CLASS_NAMES)
    raise ValueError(f"Unknown class label {value!r}; expected one of {expected}")


def load_artifacts(data_dir):
    """Load one 33-feature row and one metadata row per sample."""
    data_dir = Path(data_dir)
    features_path = data_dir / "features.npy"
    segments_path = data_dir / "segments.csv"
    if not features_path.is_file():
        raise FileNotFoundError(f"Missing prepared feature artifact: {features_path}")
    if not segments_path.is_file():
        raise FileNotFoundError(f"Missing segment metadata artifact: {segments_path}")

    features = np.asarray(np.load(features_path, allow_pickle=True), dtype=np.float32)
    if features.ndim != 2 or features.shape[1] != FEATURE_COUNT:
        raise ValueError(
            f"features.npy must have shape (N, {FEATURE_COUNT}); got {features.shape}"
        )
    if not np.all(np.isfinite(features)):
        raise ValueError("features.npy contains non-finite values")

    with segments_path.open("r", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError(f"{segments_path} must have a header row")
        label_column = _column_name(
            reader.fieldnames,
            (
                "label",
                "label_id",
                "technology",
                "technology_label",
                "tech",
                "tech_label",
                "class",
                "class_label",
                "target",
                "y",
            ),
        )
        sensor_column = _column_name(
            reader.fieldnames,
            ("sensor", "sensor_id", "id_sensor", "sensor_name", "name_sensor", "group"),
        )
        if label_column is None:
            raise ValueError(
                f"{segments_path} needs a class column named label, technology, or class"
            )
        random_split_column = _column_name(reader.fieldnames, ("random_split",))
        sensor_split_column = _column_name(reader.fieldnames, ("sensor_split",))
        if random_split_column is None or sensor_split_column is None:
            raise ValueError(
                f"{segments_path} must contain random_split and sensor_split columns"
            )

        rows = [
            row
            for row in reader
            if row and any(str(value).strip() for value in row.values() if value is not None)
        ]

    if len(rows) != len(features):
        raise ValueError(
            "features.npy and segments.csv must contain the same number of samples "
            f"({len(features)} features versus {len(rows)} metadata rows)"
        )

    labels = np.asarray([_label_id(row[label_column]) for row in rows], dtype=np.int64)
    sensors = None
    if sensor_column is not None:
        sensors = np.asarray([str(row[sensor_column]).strip() for row in rows], dtype=object)
    assignments = {
        "class_random": np.asarray(
            [str(row[random_split_column]).strip() for row in rows], dtype=object
        ),
        "sensor": np.asarray(
            [str(row[sensor_split_column]).strip() for row in rows], dtype=object
        ),
    }
    return features, labels, sensors, assignments


def _class_random_split(indices, labels, test_size, seed):
    """Split indices, stratifying whenever the requested split is feasible."""
    split_size = test_size
    unique_labels, label_counts = np.unique(labels, return_counts=True)
    if isinstance(test_size, float) and len(unique_labels) > 1 and label_counts.min() >= 2:
        requested_count = int(np.ceil(test_size * len(indices)))
        minimum_count = len(unique_labels)
        if requested_count < minimum_count and len(indices) - minimum_count >= minimum_count:
            split_size = minimum_count
    try:
        train_indices, test_indices = train_test_split(
            indices,
            test_size=split_size,
            random_state=seed,
            stratify=labels,
        )
        return train_indices, test_indices, True
    except ValueError:
        train_indices, test_indices = train_test_split(
            indices, test_size=split_size, random_state=seed
        )
        return train_indices, test_indices, False


def _validation_split(indices, labels, split, sensors, seed):
    if len(indices) < 2:
        return indices, np.empty(0, dtype=np.int64), False

    if split == "sensor" and sensors is not None:
        groups = sensors[indices]
        if len(np.unique(groups)) >= 2 and np.all(groups != ""):
            splitter = GroupShuffleSplit(
                n_splits=1, test_size=VALIDATION_SIZE, random_state=seed
            )
            train_relative, validation_relative = next(
                splitter.split(indices, labels[indices], groups)
            )
            return indices[train_relative], indices[validation_relative], True

    train_indices, validation_indices, stratified = _class_random_split(
        indices, labels[indices], VALIDATION_SIZE, seed
    )
    return train_indices, validation_indices, stratified


def make_splits(labels, sensors, assignments, split, seed):
    """Use the prepared manifest holdout and split its training rows for validation."""
    indices = np.arange(len(labels), dtype=np.int64)
    if len(indices) < 2:
        raise ValueError("At least two samples are required to create a test split")

    prepared = np.asarray(assignments[split])
    unexpected = set(np.unique(prepared)) - {"train", "test"}
    if unexpected:
        raise ValueError(f"Prepared {split} split contains invalid values: {sorted(unexpected)}")
    train_pool = indices[prepared == "train"]
    test_indices = indices[prepared == "test"]
    effective_split = split
    test_stratified = split == "class_random"
    if split == "sensor":
        if sensors is None or np.any(sensors == ""):
            raise ValueError("Prepared sensor split requires non-empty sensor metadata")
        if set(sensors[train_pool]) & set(sensors[test_indices]):
            raise RuntimeError("Prepared sensor split leaks a sensor between train and test")

    for class_id, class_name in enumerate(CLASS_NAMES):
        if not np.any(labels[train_pool] == class_id) or not np.any(labels[test_indices] == class_id):
            raise ValueError(f"Prepared {split} split does not contain {class_name} in both partitions")

    train_indices, validation_indices, validation_stratified = _validation_split(
        np.asarray(train_pool),
        labels,
        split if effective_split == "sensor" else "class_random",
        sensors,
        seed,
    )
    train_indices = np.asarray(train_indices, dtype=np.int64)
    validation_indices = np.asarray(validation_indices, dtype=np.int64)
    test_indices = np.asarray(test_indices, dtype=np.int64)

    if len(train_indices) == 0 or len(test_indices) == 0:
        raise ValueError("The requested split produced an empty train or test set")
    if np.intersect1d(train_indices, test_indices).size:
        raise RuntimeError("Train and test splits overlap")
    if np.intersect1d(validation_indices, test_indices).size:
        raise RuntimeError("Validation and test splits overlap")
    if split == "sensor":
        train_groups = set(sensors[train_indices])
        validation_groups = set(sensors[validation_indices])
        test_groups = set(sensors[test_indices])
        if train_groups & test_groups or validation_groups & test_groups:
            raise RuntimeError("Sensor split leaked a sensor between training and test")

    print(
        f"Split: {effective_split}; test stratified={test_stratified}; "
        f"validation stratified/group-aware={validation_stratified}"
    )
    return train_indices, validation_indices, test_indices


def build_autoencoder(input_dim=FEATURE_COUNT, compressed_dim=LATENT_DIM, learning_rate=LEARNING_RATE):
    """Build the 33 -> 64 -> 32 -> 16 -> 32 -> 64 -> 33 autoencoder."""
    del learning_rate  # The optimizer is created by the training loop.
    autoencoder = Autoencoder(input_dim, compressed_dim)
    return autoencoder, autoencoder.encoder


def build_lstm_classifier(num_classes=len(CLASS_NAMES), learning_rate=LEARNING_RATE):
    """Build the 16-step, one-value-per-step LSTM classifier."""
    del learning_rate  # The optimizer is created by the training loop.
    return LSTMClassifier(num_classes)


def _fit_model(
    model,
    x_train,
    y_train,
    x_validation,
    y_validation,
    epochs,
    batch_size,
    *,
    task="autoencoder",
    device=None,
    seed=0,
    model_name="model",
):
    """Train one model with Adam and validation-loss early stopping."""
    if device is None:
        device = torch.device("cpu")
    if task not in {"autoencoder", "classifier"}:
        raise ValueError(f"Unknown training task: {task}")

    x_train = np.asarray(x_train, dtype=np.float32)
    x_validation = np.asarray(x_validation, dtype=np.float32)
    target_dtype = np.int64 if task == "classifier" else np.float32
    y_train = np.asarray(y_train, dtype=target_dtype)
    y_validation = np.asarray(y_validation, dtype=target_dtype)
    train_dataset = TensorDataset(
        torch.from_numpy(x_train), torch.from_numpy(y_train)
    )
    validation_dataset = TensorDataset(
        torch.from_numpy(x_validation), torch.from_numpy(y_validation)
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    validation_loader = (
        DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        if len(validation_dataset)
        else None
    )

    model.to(device)
    criterion = nn.MSELoss() if task == "autoencoder" else nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    history = {"loss": [], "val_loss": []}
    best_state = None
    best_validation_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(epochs):
        model.train()
        training_loss = 0.0
        training_samples = 0
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(inputs), targets)
            loss.backward()
            optimizer.step()
            count = inputs.shape[0]
            training_loss += float(loss.detach().cpu()) * count
            training_samples += count
        training_loss /= max(training_samples, 1)
        history["loss"].append(training_loss)

        validation_loss = None
        if validation_loader is not None:
            model.eval()
            validation_total = 0.0
            validation_samples = 0
            with torch.no_grad():
                for inputs, targets in validation_loader:
                    inputs = inputs.to(device)
                    targets = targets.to(device)
                    loss = criterion(model(inputs), targets)
                    count = inputs.shape[0]
                    validation_total += float(loss.cpu()) * count
                    validation_samples += count
            validation_loss = validation_total / max(validation_samples, 1)
            history["val_loss"].append(validation_loss)
            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                best_epoch = epoch + 1
                epochs_without_improvement = 0
                best_state = copy.deepcopy(
                    {key: value.detach().cpu() for key, value in model.state_dict().items()}
                )
            else:
                epochs_without_improvement += 1

        print(
            f"{model_name} epoch {epoch + 1}/{epochs}: "
            f"loss={training_loss:.6f}"
            + (f" val_loss={validation_loss:.6f}" if validation_loss is not None else "")
        )
        if validation_loader is not None and epochs_without_improvement >= PATIENCE:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    history["best_epoch"] = best_epoch if best_state is not None else len(history["loss"])
    history["epochs_completed"] = len(history["loss"])
    return history


def _encode_rows(encoder, features, batch_size, device):
    dataset = TensorDataset(torch.from_numpy(np.asarray(features, dtype=np.float32)))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    encoder.eval()
    encoded_parts = []
    with torch.no_grad():
        for (batch,) in loader:
            encoded_parts.append(encoder(batch.to(device)).cpu().numpy())
    return np.concatenate(encoded_parts, axis=0) if encoded_parts else np.empty((0, LATENT_DIM))


def _write_matrix(path, matrix, normalized):
    with Path(path).open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["true\\pred"] + CLASS_NAMES)
        for label, row in zip(CLASS_NAMES, matrix):
            values = [float(value) for value in row] if normalized else [int(value) for value in row]
            writer.writerow([label] + values)


def evaluate(y_true, y_pred, output_dir):
    labels = np.arange(len(CLASS_NAMES))
    counts = confusion_matrix(y_true, y_pred, labels=labels)
    row_totals = counts.sum(axis=1, keepdims=True)
    normalized = np.divide(
        counts.astype(np.float64),
        row_totals,
        out=np.zeros_like(counts, dtype=np.float64),
        where=row_totals != 0,
    )
    report_text = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=CLASS_NAMES,
        zero_division=0,
    )
    report_dict = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=CLASS_NAMES,
        zero_division=0,
        output_dict=True,
    )
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
        "classification_report": report_dict,
    }

    print("\nInteger confusion matrix:")
    print(counts)
    print("\nNormalized confusion matrix:")
    print(np.array2string(normalized, formatter={"float_kind": lambda value: f"{value:.4f}"}))
    print("\nClassification report:")
    print(report_text)
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Balanced accuracy: {metrics['balanced_accuracy']:.4f}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")

    _write_matrix(output_dir / "confusion_matrix_integer.csv", counts, normalized=False)
    _write_matrix(output_dir / "confusion_matrix_normalized.csv", normalized, normalized=True)
    (output_dir / "classification_report.txt").write_text(report_text)
    return counts, normalized, metrics


def _save_predictions(output_dir, indices, y_true, y_pred, probabilities):
    with (output_dir / "test_predictions.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["row_index", "true_label", "predicted_label"] + CLASS_NAMES)
        for index, truth, prediction, probability in zip(
            indices, y_true, y_pred, probabilities
        ):
            writer.writerow(
                [int(index), CLASS_NAMES[int(truth)], CLASS_NAMES[int(prediction)]]
                + [float(value) for value in probability]
            )


def _cpu_state_dict(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _architecture_metadata():
    return {
        "feature_count": FEATURE_COUNT,
        "latent_dim": LATENT_DIM,
        "class_names": CLASS_NAMES,
        "autoencoder": {
            "encoder": [FEATURE_COUNT, 64, 32, LATENT_DIM],
            "decoder": [LATENT_DIM, 32, 64, FEATURE_COUNT],
            "hidden_activation": "ReLU",
            "output_activation": "linear",
        },
        "classifier": {
            "input_shape": [LATENT_DIM, 1],
            "lstm_hidden_sizes": [32, 16],
            "dense": [16, len(CLASS_NAMES)],
            "dense_activation": "ReLU",
            "dropout": DROPOUT_RATE,
            "output": "six logits; softmax applied for prediction",
        },
    }


def _save_checkpoint(
    path,
    autoencoder,
    classifier,
    split_indices,
    predictions,
    confusion_matrices,
    metrics,
    training_metadata,
):
    checkpoint = {
        "format_version": 1,
        "architecture": _architecture_metadata(),
        "model_state_dicts": {
            "autoencoder": _cpu_state_dict(autoencoder),
            "classifier": _cpu_state_dict(classifier),
        },
        "split_indices": {
            key: torch.as_tensor(value, dtype=torch.int64)
            for key, value in split_indices.items()
        },
        "predictions": {
            key: torch.as_tensor(value)
            for key, value in predictions.items()
        },
        "confusion_matrices": {
            "integer": torch.as_tensor(confusion_matrices["integer"], dtype=torch.int64),
            "normalized": torch.as_tensor(confusion_matrices["normalized"], dtype=torch.float64),
        },
        "metrics": metrics,
        "training": training_metadata,
    }
    torch.save(checkpoint, path)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train the prepared-feature AE/LSTM pipeline")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("training_output"))
    parser.add_argument(
        "--split", choices=("class_random", "sensor"), default="class_random"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--smoke", action="store_true", help=f"Run at most {SMOKE_EPOCHS} epochs")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError("--epochs and --batch-size must be positive")

    set_random_seed(args.seed)
    device = select_device(args.device)
    epochs = min(args.epochs, SMOKE_EPOCHS) if args.smoke else args.epochs
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Using device: {device}")
    print("Loading prepared artifacts...")
    features, labels, sensors, assignments = load_artifacts(args.data_dir)
    print(f"Features: {features.shape}; samples: one row each")
    train_indices, validation_indices, test_indices = make_splits(
        labels, sensors, assignments, args.split, args.seed
    )
    manifest_train_indices = np.flatnonzero(assignments[args.split] == "train").astype(np.int64)
    manifest_test_indices = np.flatnonzero(assignments[args.split] == "test").astype(np.int64)
    print(
        f"Train: {len(train_indices)}, validation: {len(validation_indices)}, "
        f"test: {len(test_indices)}"
    )

    # The scaler is fitted exclusively on rows used for model training.
    scaler = MinMaxScaler()
    scaler.fit(features[train_indices])
    scaled_features = scaler.transform(features).astype(np.float32)

    print("\nTraining autoencoder...")
    autoencoder, encoder = build_autoencoder()
    ae_history = _fit_model(
        autoencoder,
        scaled_features[train_indices],
        scaled_features[train_indices],
        scaled_features[validation_indices],
        scaled_features[validation_indices],
        epochs,
        args.batch_size,
        task="autoencoder",
        device=device,
        seed=args.seed,
        model_name="autoencoder",
    )

    print("Encoding each row...")
    encoded = _encode_rows(encoder, scaled_features, args.batch_size, device)
    if encoded.shape != (len(features), LATENT_DIM):
        raise RuntimeError(f"Unexpected encoded shape: {encoded.shape}")
    lstm_features = encoded.reshape(len(encoded), LATENT_DIM, 1).astype(np.float32)
    print(f"Encoded shape: {encoded.shape}; LSTM input: {lstm_features.shape}")

    print("\nTraining LSTM classifier...")
    classifier = build_lstm_classifier()
    lstm_history = _fit_model(
        classifier,
        lstm_features[train_indices],
        labels[train_indices],
        lstm_features[validation_indices],
        labels[validation_indices],
        epochs,
        args.batch_size,
        task="classifier",
        device=device,
        seed=args.seed + 1,
        model_name="classifier",
    )

    print("\nEvaluating test rows without averaging...")
    classifier.eval()
    test_dataset = TensorDataset(torch.from_numpy(lstm_features[test_indices]))
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    probability_parts = []
    with torch.no_grad():
        for (batch,) in test_loader:
            logits = classifier(batch.to(device))
            probability_parts.append(torch.softmax(logits, dim=1).cpu().numpy())
    probabilities = np.concatenate(probability_parts, axis=0)
    if probabilities.shape != (len(test_indices), len(CLASS_NAMES)):
        raise RuntimeError(f"Unexpected probability shape: {probabilities.shape}")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-4):
        raise RuntimeError("Classifier probabilities do not sum to one")
    print(f"Probability shape: {probabilities.shape}")
    y_true = labels[test_indices]
    y_pred = np.argmax(probabilities, axis=1).astype(np.int64)
    counts, normalized, metrics = evaluate(y_true, y_pred, args.output_dir)
    _save_predictions(args.output_dir, test_indices, y_true, y_pred, probabilities)

    label_encoder = LabelEncoder().fit(CLASS_NAMES)
    joblib.dump(scaler, args.output_dir / "scaler.pkl")
    joblib.dump(label_encoder, args.output_dir / "label_encoder.pkl")
    split_indices = {
        "train": train_indices,
        "validation": validation_indices,
        "test": test_indices,
        "manifest_train": manifest_train_indices,
        "manifest_test": manifest_test_indices,
    }
    np.savez(args.output_dir / "split_indices.npz", **split_indices)
    history = {"autoencoder": ae_history, "lstm": lstm_history}
    (args.output_dir / "training_history.json").write_text(json.dumps(history, indent=2))
    training_metadata = {
        "split": args.split,
        "seed": args.seed,
        "epochs_requested": args.epochs,
        "epochs": epochs,
        "batch_size": args.batch_size,
        "smoke": args.smoke,
        "device": str(device),
        "learning_rate": LEARNING_RATE,
        "early_stopping_patience": PATIENCE,
        "train_samples": int(len(train_indices)),
        "validation_samples": int(len(validation_indices)),
        "test_samples": int(len(test_indices)),
        "manifest_train_samples": int(len(manifest_train_indices)),
        "manifest_test_samples": int(len(manifest_test_indices)),
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps({**training_metadata, **metrics}, indent=2)
    )
    _save_checkpoint(
        args.output_dir / "training_checkpoint.pt",
        autoencoder,
        classifier,
        split_indices,
        {
            "row_indices": test_indices,
            "true_labels": y_true,
            "predicted_labels": y_pred,
            "probabilities": probabilities,
        },
        {"integer": counts, "normalized": normalized},
        metrics,
        training_metadata,
    )
    print(f"\nSaved training artifacts under {args.output_dir}")


if __name__ == "__main__":
    main()
