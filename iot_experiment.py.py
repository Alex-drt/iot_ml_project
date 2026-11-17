from __future__ import annotations

import os

# Должно быть установлено до инициализации CUDA.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import hashlib
import importlib.metadata
import json
import platform
import random
import subprocess
import sys
import time
import warnings
from decimal import Decimal, InvalidOperation
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


FORBIDDEN_FEATURES = {
    "target",
    "label",
    "class_label",
    "machine failure",
    "twf",
    "hdf",
    "pwf",
    "osf",
    "rnf",
    "in_study_repair",
    "length_of_study_time_step",
    "time_to_event",
    "tte",
    "rul",
    "attack",
    "attack_type",
    "attack_name",
    "vehicle_id",
    "udi",
    "product id",
    "sample_id",
    "sequence_id",
    "source_id",
    "filename",
    "y_true",
    "y_pred",
}

SCANIA_COST_MATRIX = np.asarray(
    [
        [0, 7, 8, 9, 10],
        [200, 0, 7, 8, 9],
        [300, 200, 0, 7, 8],
        [400, 300, 200, 0, 7],
        [500, 400, 300, 200, 0],
    ],
    dtype=np.int64,
)


def json_default(value):
    """Преобразование служебных типов без превращения чисел в строки."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(
        f"Object of type {type(value).__name__} is not JSON serializable"
    )


def save_json(path, obj):
    Path(path).write_text(
        json.dumps(
            obj,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
            default=json_default,
        ),
        encoding="utf-8",
    )


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(
            lambda: stream.read(1024 * 1024),
            b"",
        ):
            digest.update(block)
    return digest.hexdigest()


def environment():
    versions = {}
    for name in [
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "joblib",
        "torch",
    ]:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None

    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": versions,
        "cublas_workspace_config": os.environ.get(
            "CUBLAS_WORKSPACE_CONFIG"
        ),
    }


def require_finite(name, values):
    values = np.asarray(values)
    if not np.isfinite(values).all():
        raise ValueError(
            f"{name}: non-finite values detected "
            "(NaN, +inf or -inf)"
        )


def load_data(args):
    feature_path = Path(args.features)
    features = json.loads(
        feature_path.read_text(encoding="utf-8")
    )

    if (
        not isinstance(features, list)
        or not features
        or not all(
            isinstance(name, str)
            and name.strip()
            and name == name.strip()
            for name in features
        )
        or len({name.casefold() for name in features}) != len(features)
    ):
        raise ValueError(
            "features.json must contain a non-empty list "
            "of unique column names"
        )

    excluded = {
        str(name).strip().lower()
        for name in [
            args.target,
            args.id,
            args.group,
            args.time,
        ]
        if name is not None
    }

    for name in features:
        normalized = name.strip().lower()
        if (
            normalized in excluded
            or normalized in FORBIDDEN_FEATURES
        ):
            raise ValueError(f"Forbidden feature: {name}")

    frames = {}
    preparation_path = feature_path.parent / "preparation.json"
    preparation_metadata = None
    if preparation_path.is_file():
        def reject_nonstandard_constant(value):
            raise ValueError(
                f"Invalid non-finite value {value} in {preparation_path}"
            )

        preparation_metadata = json.loads(
            preparation_path.read_text(encoding="utf-8"),
            parse_constant=reject_nonstandard_constant,
        )

    audit: dict[str, object] = {
        "features_file": {
            "path": str(feature_path.resolve()),
            "sha256": sha256(feature_path),
        }
    }
    audit["preparation_metadata"] = (
        {
            "path": str(preparation_path.resolve()),
            "sha256": sha256(preparation_path),
            "metadata": preparation_metadata,
        }
        if preparation_metadata is not None
        else None
    )
    paths = {
        "train": args.train,
        "validation": args.validation,
        "test": args.test,
    }

    for split, path in paths.items():
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(path)

        dtype = {args.id: str, args.target: str}
        if args.group:
            dtype[args.group] = str

        frame = pd.read_csv(
            path,
            dtype=dtype,
        ).reset_index(drop=True)

        required = features + [args.target, args.id]
        if args.group:
            required.append(args.group)
        if args.time:
            required.append(args.time)

        missing = sorted(set(required) - set(frame.columns))
        if missing:
            raise ValueError(
                f"{split}: missing columns: {missing}"
            )
        if frame.empty:
            raise ValueError(f"{split}: empty dataset")

        sample_ids = frame[args.id]
        if sample_ids.isna().any() or sample_ids.str.strip().eq("").any():
            raise ValueError(
                f"{split}: sample IDs must be non-empty"
            )
        if sample_ids.str.strip().ne(sample_ids).any():
            raise ValueError(
                f"{split}: sample IDs cannot have surrounding whitespace"
            )
        if sample_ids.duplicated().any():
            raise ValueError(f"{split}: duplicate sample ID")

        if args.group:
            group_ids = frame[args.group]
            if group_ids.isna().any() or group_ids.str.strip().eq("").any():
                raise ValueError(
                    f"{split}: group IDs must be non-empty"
                )
            if group_ids.str.strip().ne(group_ids).any():
                raise ValueError(
                    f"{split}: group IDs cannot have surrounding whitespace"
                )

        frame[args.target] = validated_labels(
            f"{split}: targets",
            frame[args.target].to_numpy(),
            2**63,
        )

        for name in features:
            frame[name] = pd.to_numeric(
                frame[name],
                errors="raise",
            )

        raw_features = frame[features].to_numpy(
            dtype=np.float64
        )
        infinity_count = int(np.isinf(raw_features).sum())

        frame[features] = frame[features].replace(
            [np.inf, -np.inf],
            np.nan,
        )

        if infinity_count:
            warnings.warn(
                f"{split}: {infinity_count} infinite feature "
                "values replaced with missing values",
                stacklevel=2,
            )

        if args.time:
            frame[args.time] = pd.to_numeric(
                frame[args.time],
                errors="raise",
            )
            require_finite(
                f"{split}: time",
                frame[args.time].to_numpy(dtype=float),
            )

        frames[split] = frame
        audit[split] = {
            "path": str(path.resolve()),
            "sha256": sha256(path),
            "rows": int(len(frame)),
            "classes": {
                str(int(key)): int(value)
                for key, value in (
                    frame[args.target].value_counts().items()
                )
            },
            "infinite_feature_values_replaced": infinity_count,
            "missing_feature_values": int(
                frame[features].isna().sum().sum()
            ),
        }

    classes = sorted(
        int(value)
        for value in frames["train"][args.target].unique()
    )

    if args.model == "lstm_ae":
        if classes not in ([0], [0, 1]):
            raise ValueError(
                "LSTM AE requires normal=0 and anomaly=1; "
                "training must contain normal observations"
            )
        classes = [0, 1]
    elif (
        len(classes) < 2
        or classes != list(range(len(classes)))
    ):
        raise ValueError(
            "Training classes must be contiguous: 0,1,...,K-1"
        )

    for split, frame in frames.items():
        if not set(frame[args.target]).issubset(classes):
            raise ValueError(
                f"{split}: class is absent from the allowed "
                "training class set"
            )

    if args.model != "lstm_ae" and len(classes) == 2:
        if set(frames["validation"][args.target]) != {0, 1}:
            raise ValueError(
                "Binary F1 threshold selection requires both "
                "classes in validation"
            )

    pairs = [
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    ]

    for first, second in pairs:
        for column in [args.id, args.group]:
            if column is None:
                continue
            overlap = (
                set(frames[first][column])
                & set(frames[second][column])
            )
            if overlap:
                raise ValueError(
                    f"{column} overlaps: {first}/{second}; "
                    f"number of overlapping values: {len(overlap)}"
                )

    audit["feature_overlap"] = {}

    feature_hashes = {
        split: set(
            pd.util.hash_pandas_object(
                frame[features],
                index=False,
            ).tolist()
        )
        for split, frame in frames.items()
    }

    for first, second in pairs:
        audit["feature_overlap"][f"{first}__{second}"] = len(
            feature_hashes[first] & feature_hashes[second]
        )

    audit["feature_overlap_note"] = (
        "Counts concern unique feature hashes, not the number "
        "of repeated rows. Hash collisions are theoretically "
        "possible. Overlap is reported, not automatically removed."
    )
    audit["independence_note"] = (
        "Different sample IDs do not prove independence. "
        "Group meanings and source provenance must be verified."
    )
    audit["n_classes"] = len(classes)

    return frames, features, len(classes), audit


def build_window_index(frame, target, args):
    """Индексы окон без копирования всех окон в память."""
    group_column = args.group
    time_column = args.time
    if (
        not isinstance(group_column, str)
        or not group_column.strip()
        or not isinstance(time_column, str)
        or not time_column.strip()
    ):
        raise ValueError(
            "LSTM requires both --group and --time"
        )
    if (
        not isinstance(args.window, int)
        or isinstance(args.window, bool)
        or args.window < 2
    ):
        raise ValueError("window must be an integer of at least two")
    target = np.asarray(target)
    if target.ndim != 1 or len(target) != len(frame):
        raise ValueError("Window labels must align with input rows")

    clock = frame[time_column].to_numpy(dtype=float)
    require_finite("Window time", clock)

    order_parts = []
    start_parts = []
    offset = 0

    for positions in frame.groupby(
        group_column,
        sort=False,
    ).indices.values():
        positions = np.asarray(positions, dtype=np.int64)
        positions = positions[
            np.argsort(clock[positions], kind="stable")
        ]

        differences = np.diff(clock[positions])

        if np.any(differences <= 0):
            raise ValueError(
                "Time must strictly increase inside each group; "
                "duplicate timestamps are not allowed"
            )

        boundaries = [0]
        if args.max_gap is not None:
            boundaries.extend(
                (np.flatnonzero(
                    differences > args.max_gap
                ) + 1).tolist()
            )
        boundaries.append(len(positions))

        order_parts.append(positions)

        for left, right in zip(
            boundaries[:-1],
            boundaries[1:],
        ):
            if right - left >= args.window:
                if getattr(args, "endpoint_only_evaluation", False):
                    if right == len(positions):
                        start_parts.append(
                            np.asarray(
                                [offset + right - args.window],
                                dtype=np.int64,
                            )
                        )
                else:
                    start_parts.append(
                        np.arange(
                            offset + left,
                            offset + right - args.window + 1,
                            dtype=np.int64,
                        )
                    )

        offset += len(positions)

    if not start_parts:
        raise ValueError(
            "No complete LSTM windows. Check window length, "
            "group sizes and --max-gap"
        )

    order = np.concatenate(order_parts)
    starts = np.concatenate(start_parts)
    endpoints = order[starts + args.window - 1]

    abnormal = (target[order] != 0).astype(np.int64)
    cumulative = np.empty(len(abnormal) + 1, dtype=np.int64)
    cumulative[0] = 0
    np.cumsum(abnormal, out=cumulative[1:])

    clean = (
        cumulative[starts + args.window]
        - cumulative[starts]
    ) == 0
    clean_before_limit = clean.copy()

    max_windows = getattr(
        args,
        "max_train_windows_per_group",
        None,
    )
    if max_windows is not None:
        endpoint_groups = frame[group_column].to_numpy()[endpoints]
        selected = np.zeros(len(starts), dtype=bool)
        for group in pd.unique(endpoint_groups):
            candidates = np.flatnonzero(
                clean_before_limit & (endpoint_groups == group)
            )
            if len(candidates) > max_windows:
                positions = np.linspace(
                    0,
                    len(candidates) - 1,
                    num=max_windows,
                    dtype=np.int64,
                )
                candidates = candidates[positions]
            selected[candidates] = True
        clean = selected

    return {
        "order": order,
        "starts": starts,
        "endpoints": endpoints,
        "clean": clean,
        "fully_normal_windows_before_limit": int(
            clean_before_limit.sum()
        ),
    }


def window_index_arguments(args, split):
    window_args = argparse.Namespace(**vars(args))
    window_args.endpoint_only_evaluation = (
        getattr(args, "endpoint_only_evaluation", False)
        and split != "train"
    )
    window_args.max_train_windows_per_group = (
        getattr(args, "max_train_windows_per_group", None)
        if split == "train"
        else None
    )
    return window_args


class LazyWindowView:
    def __init__(self, values, specification, window, mask=None):
        self.values = values
        self.order = specification["order"]
        self.window = window
        starts = specification["starts"]
        if mask is not None:
            mask = np.asarray(mask, dtype=bool)
            if mask.shape != starts.shape:
                raise ValueError("Window mask does not match window index")
            starts = starts[mask]
        self.starts = starts

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, index):
        start = int(self.starts[index])
        rows = self.order[start : start + self.window]
        return self.values[rows]


def select_threshold(target, score):
    require_finite("Validation scores", score)

    target = np.asarray(target)
    if target.ndim != 1 or len(np.unique(target)) != 2:
        raise ValueError(
            "F1 threshold selection requires both validation classes"
        )

    precision, recall, thresholds = precision_recall_curve(
        target,
        score,
    )

    if not len(thresholds):
        raise ValueError("No thresholds available")

    denominator = precision[:-1] + recall[:-1]

    values = np.divide(
        2 * precision[:-1] * recall[:-1],
        denominator,
        out=np.zeros_like(thresholds),
        where=denominator != 0,
    )

    # При равной F1 выбирается больший порог.
    best = np.flatnonzero(values == values.max())[-1]

    return float(thresholds[best])


def select_balanced_accuracy_threshold(target, score):
    target = np.asarray(target)
    if target.ndim != 1 or len(np.unique(target)) != 2:
        raise ValueError(
            "Balanced-accuracy threshold selection requires both "
            "validation classes"
        )
    score = np.asarray(score, dtype=float)
    if score.shape != target.shape:
        raise ValueError(
            "Validation scores must align with binary targets"
        )
    require_finite("Validation scores", score)

    false_positive_rate, true_positive_rate, thresholds = roc_curve(
        target,
        score,
    )
    finite = np.isfinite(thresholds)
    balanced_accuracy = 0.5 * (
        true_positive_rate[finite] + 1 - false_positive_rate[finite]
    )
    candidates = thresholds[finite]
    best = np.flatnonzero(
        balanced_accuracy == balanced_accuracy.max()
    )
    return float(candidates[best].max())


def scania_cost_decision(scores):
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 2 or scores.shape[1] != len(SCANIA_COST_MATRIX):
        raise ValueError(
            "SCANIA cost decisions require one score per five classes"
        )
    require_finite("SCANIA class scores", scores)
    if np.any(scores < 0):
        raise ValueError("SCANIA class scores must be non-negative")
    expected_costs = scores @ SCANIA_COST_MATRIX
    return np.argmin(expected_costs, axis=1)


def validated_labels(name, values, n_classes):
    raw = np.asarray(values)
    if raw.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional label array")

    labels = []
    for value in raw:
        try:
            numeric = Decimal(str(value).strip())
        except (InvalidOperation, TypeError, ValueError) as exc:
            raise ValueError(
                f"{name} must contain integer class labels"
            ) from exc
        if (
            not numeric.is_finite()
            or numeric != numeric.to_integral_value()
            or numeric < 0
            or numeric >= n_classes
        ):
            raise ValueError(
                f"{name} must contain integer labels in "
                f"[0, {n_classes - 1}]"
            )
        labels.append(int(numeric))

    return np.asarray(labels, dtype=np.int64)


def metrics(target, prediction, score, n_classes, scania=False):
    if (
        not isinstance(n_classes, (int, np.integer))
        or isinstance(n_classes, (bool, np.bool_))
        or n_classes < 2
    ):
        raise ValueError("n_classes must be an integer of at least two")
    if scania and n_classes != len(SCANIA_COST_MATRIX):
        raise ValueError("SCANIA cost metrics require exactly five classes")

    target = validated_labels("target", target, n_classes)
    prediction = validated_labels("prediction", prediction, n_classes)
    if len(target) == 0 or len(target) != len(prediction):
        raise ValueError("Invalid evaluation array lengths")

    labels = np.arange(n_classes)

    matrix = confusion_matrix(
        target,
        prediction,
        labels=labels,
    )

    support = matrix.sum(axis=1)
    present = support > 0
    balanced = np.mean(
        np.diag(matrix)[present] / support[present]
    )
    true_positive = np.diag(matrix)
    predicted_support = matrix.sum(axis=0)
    class_precision = np.divide(
        true_positive,
        predicted_support,
        out=np.zeros(n_classes, dtype=float),
        where=predicted_support != 0,
    )
    class_recall = np.divide(
        true_positive,
        support,
        out=np.zeros(n_classes, dtype=float),
        where=support != 0,
    )
    class_f1 = np.divide(
        2 * class_precision * class_recall,
        class_precision + class_recall,
        out=np.zeros(n_classes, dtype=float),
        where=(class_precision + class_recall) != 0,
    )
    class_support = support

    result = {
        "n_test": int(len(target)),
        "accuracy": float(
            accuracy_score(target, prediction)
        ),
        "balanced_accuracy": float(balanced),
        "macro_f1": float(
            f1_score(
                target,
                prediction,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        ),
        "confusion_matrix": matrix.tolist(),
        "per_class": {
            str(index): {
                "precision": float(class_precision[index]),
                "recall": float(class_recall[index]),
                "f1": float(class_f1[index]),
                "support": int(class_support[index]),
            }
            for index in range(n_classes)
        },
    }

    if scania:
        total_cost = int(
            np.sum(SCANIA_COST_MATRIX[target, prediction])
        )
        result.update(
            total_cost=total_cost,
            mean_cost=float(total_cost / len(target)),
        )

    if n_classes == 2:
        if score is None:
            raise ValueError("Binary ranking metrics require a score array")
        score = np.asarray(score, dtype=float)
        if score.shape != (len(target),):
            raise ValueError(
                "Binary scores must have one value per evaluation row"
            )
        require_finite("Evaluation scores", score)

        tn, fp, fn, tp = map(int, matrix.ravel())
        both_classes = len(np.unique(target)) == 2
        undefined_metric_reasons = {}

        if not both_classes:
            present_class = int(target[0])
            reason = (
                "The test split contains only class "
                f"{present_class}; ranking discrimination cannot be "
                "assessed on both classes."
            )
            undefined_metric_reasons["roc_auc"] = reason
            undefined_metric_reasons["average_precision"] = reason
        if fp + tn == 0:
            undefined_metric_reasons["fpr"] = (
                "FPR is undefined because the test split contains "
                "no negative samples."
            )

        result.update(
            tn=tn,
            fp=fp,
            fn=fn,
            tp=tp,
            precision=float(
                precision_score(
                    target,
                    prediction,
                    zero_division=0,
                )
            ),
            recall=float(
                recall_score(
                    target,
                    prediction,
                    zero_division=0,
                )
            ),
            f1=float(
                f1_score(
                    target,
                    prediction,
                    zero_division=0,
                )
            ),
            fpr=(
                float(fp / (fp + tn))
                if fp + tn else None
            ),
            alarm_rate=float(
                np.mean(prediction == 1)
            ),
            prevalence=float(
                np.mean(target == 1)
            ),
        )

        result["roc_auc"] = (
            float(roc_auc_score(target, score))
            if both_classes else None
        )
        result["average_precision"] = (
            float(average_precision_score(target, score))
            if both_classes else None
        )
        result["undefined_metric_reasons"] = undefined_metric_reasons

    return result





def neural_model(
    args,
    features_array,
    targets,
    n_classes,
    frames,
    seed,
    folder,
    feature_names,
):
    try:
        import torch
        from torch import nn
        from torch.utils.data import (
            DataLoader,
            Dataset,
            TensorDataset,
        )
    except ImportError as exc:
        if args.device == "cuda":
            raise RuntimeError(
                "CUDA was requested, but PyTorch is not installed"
            ) from exc
        raise RuntimeError(
            "PyTorch is required for mlp and lstm_ae"
        ) from exc

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA requested but unavailable in this environment"
        )

    use_cuda = (
        args.device == "cuda"
        or (
            args.device == "auto"
            and torch.cuda.is_available()
        )
    )
    device = torch.device("cuda" if use_cuda else "cpu")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if use_cuda:
        torch.cuda.manual_seed_all(seed)

    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    class WindowDataset(Dataset):
        def __init__(self, values, specification, window, mask=None):
            self.windows = LazyWindowView(
                values,
                specification,
                window,
                mask,
            )

        def __len__(self):
            return len(self.windows)

        def __getitem__(self, index):
            return (torch.from_numpy(self.windows[index]),)

    class Autoencoder(nn.Module):
        def __init__(self, n_features):
            super().__init__()

            self.encoder = nn.LSTM(
                n_features,
                args.hidden,
                batch_first=True,
            )
            self.to_z = nn.Linear(
                args.hidden,
                args.latent,
            )
            self.from_z = nn.Linear(
                args.latent,
                args.hidden,
            )
            self.decoder = nn.LSTM(
                args.hidden,
                args.hidden,
                batch_first=True,
            )
            self.output = nn.Linear(
                args.hidden,
                n_features,
            )

        def forward(self, inputs):
            _, (hidden, _) = self.encoder(inputs)
            latent = self.to_z(hidden[-1])

            repeated = (
                self.from_z(latent)
                .unsqueeze(1)
                .expand(-1, inputs.shape[1], -1)
            )

            decoded, _ = self.decoder(repeated)
            return self.output(decoded)

    def synchronize():
        if use_cuda:
            torch.cuda.synchronize(device)

    indices = {
        split: np.arange(len(target))
        for split, target in targets.items()
    }

    is_autoencoder = args.model == "lstm_ae"
    class_weights = None
    clean_validation = None
    window_audit = None

    if is_autoencoder:
        if n_classes != 2:
            raise ValueError(
                "Autoencoder supports normal=0, anomaly=1"
            )

        specifications = {}
        for split in features_array:
            specifications[split] = build_window_index(
                frames[split],
                targets[split],
                window_index_arguments(args, split),
            )

        indices = {
            split: specification["endpoints"]
            for split, specification in specifications.items()
        }

        train_clean = specifications["train"]["clean"]
        clean_validation = specifications["validation"]["clean"]

        if not train_clean.any():
            raise ValueError(
                "No fully normal training windows"
            )
        if not clean_validation.any():
            raise ValueError(
                "No fully normal validation windows "
                "for threshold calibration"
            )

        train_dataset = WindowDataset(
            features_array["train"],
            specifications["train"],
            args.window,
            mask=train_clean,
        )
        validation_dataset = WindowDataset(
            features_array["validation"],
            specifications["validation"],
            args.window,
        )
        validation_normal_dataset = WindowDataset(
            features_array["validation"],
            specifications["validation"],
            args.window,
            mask=clean_validation,
        )
        test_dataset = WindowDataset(
            features_array["test"],
            specifications["test"],
            args.window,
        )

        window_audit = {
            split: {
                "all_windows": int(len(specification["starts"])),
                "fully_normal_windows": int(
                    specification["clean"].sum()
                ),
                "fully_normal_windows_before_limit": int(
                    specification["fully_normal_windows_before_limit"]
                ),
            }
            for split, specification in specifications.items()
        }

        model = Autoencoder(
            features_array["train"].shape[1]
        ).to(device)

        loss_function = nn.MSELoss()

    else:
        model = nn.Sequential(
            nn.Linear(
                features_array["train"].shape[1],
                args.hidden,
            ),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(args.hidden, args.hidden),
            nn.ReLU(),
            nn.Linear(args.hidden, n_classes),
        ).to(device)

        counts = np.bincount(
            targets["train"],
            minlength=n_classes,
        )
        weights = (
            len(targets["train"])
            / (n_classes * counts)
        )

        class_weights = torch.tensor(
            weights,
            dtype=torch.float32,
            device=device,
        )

        loss_function = nn.CrossEntropyLoss(
            weight=class_weights
        )

        train_dataset = TensorDataset(
            torch.from_numpy(features_array["train"]),
            torch.from_numpy(
                targets["train"].astype(np.int64)
            ),
        )
        validation_dataset = TensorDataset(
            torch.from_numpy(features_array["validation"]),
            torch.from_numpy(
                targets["validation"].astype(np.int64)
            ),
        )
        test_dataset = TensorDataset(
            torch.from_numpy(features_array["test"])
        )

    generator = torch.Generator().manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch,
        shuffle=False,
        num_workers=0,
    )
    validation_normal_loader = (
        DataLoader(
            validation_normal_dataset,
            batch_size=args.batch,
            shuffle=False,
            num_workers=0,
        )
        if is_autoencoder
        else None
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
    )

    history = []
    best_state = None
    best_validation_loss = float("inf")
    best_epoch = None

    synchronize()
    started = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        model.train()

        loss_numerator = 0.0
        loss_denominator = 0.0

        for batch in train_loader:
            inputs = batch[0].to(device)
            optimizer.zero_grad(set_to_none=True)

            outputs = model(inputs)

            if is_autoencoder:
                loss = loss_function(outputs, inputs)
                denominator = float(inputs.numel())
            else:
                batch_targets = batch[1].to(device)
                if class_weights is None:
                    raise RuntimeError(
                        "Classifier class weights were not initialized"
                    )
                loss = loss_function(outputs, batch_targets)
                denominator = float(
                    class_weights[batch_targets].sum().detach().cpu()
                )

            if not bool(torch.isfinite(loss).item()):
                raise RuntimeError(
                    f"Non-finite training loss at epoch {epoch}"
                )

            loss.backward()

            nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=5.0,
                error_if_nonfinite=True,
            )

            optimizer.step()

            loss_numerator += (
                float(loss.detach().cpu()) * denominator
            )
            loss_denominator += denominator

        record = {
            "epoch": epoch,
            "train_loss": (
                loss_numerator / loss_denominator
            ),
        }

        if not is_autoencoder:
            model.eval()
            validation_sum = 0.0

            with torch.inference_mode():
                for batch in validation_loader:
                    inputs = batch[0].to(device)
                    batch_targets = batch[1].to(device)

                    validation_sum += float(
                        nn.functional.cross_entropy(
                            model(inputs),
                            batch_targets,
                            reduction="sum",
                        ).cpu()
                    )

            validation_loss = (
                validation_sum
                / len(validation_dataset)
            )

            if not np.isfinite(validation_loss):
                raise RuntimeError(
                    f"Non-finite validation loss at epoch {epoch}"
                )

            record["validation_loss_unweighted"] = (
                validation_loss
            )

            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                best_epoch = epoch
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                }
        else:
            if validation_normal_loader is None:
                raise RuntimeError(
                    "Normal validation loader was not initialized"
                )
            model.eval()
            validation_sum = 0.0
            validation_denominator = 0

            with torch.inference_mode():
                for batch in validation_normal_loader:
                    inputs = batch[0].to(device)
                    validation_sum += float(
                        nn.functional.mse_loss(
                            model(inputs),
                            inputs,
                            reduction="sum",
                        ).cpu()
                    )
                    validation_denominator += inputs.numel()

            validation_loss = (
                validation_sum / validation_denominator
            )
            if not np.isfinite(validation_loss):
                raise RuntimeError(
                    f"Non-finite validation loss at epoch {epoch}"
                )
            record["validation_loss_normal"] = validation_loss

            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                best_epoch = epoch
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                }

        history.append(record)

        print(
            json.dumps(
                {
                    "seed": seed,
                    **record,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    synchronize()
    fit_seconds = time.perf_counter() - started

    model.eval()

    def predict(dataset):
        loader = DataLoader(
            dataset,
            batch_size=args.batch,
            shuffle=False,
            num_workers=0,
        )

        predictions = []

        with torch.inference_mode():
            for batch in loader:
                inputs = batch[0].to(device)
                outputs = model(inputs)

                if is_autoencoder:
                    scores = (
                        (outputs - inputs) ** 2
                    ).mean(dim=(1, 2))
                else:
                    scores = torch.softmax(
                        outputs,
                        dim=1,
                    )

                predictions.append(
                    scores.cpu().numpy()
                )

        if not predictions:
            raise ValueError("Empty prediction dataset")

        result = np.concatenate(predictions)
        require_finite("Neural model scores", result)
        return result

    validation_scores = predict(validation_dataset)

    synchronize()
    started = time.perf_counter()

    test_scores = predict(test_dataset)

    synchronize()
    inference_seconds = time.perf_counter() - started

    torch.save(
        {
            "state_dict": {
                name: value.detach().cpu()
                for name, value in model.state_dict().items()
            },
            "model": args.model,
            "features": feature_names,
            "n_features": features_array["train"].shape[1],
            "n_classes": n_classes,
            "hidden": args.hidden,
            "latent": args.latent,
            "window": args.window,
            "max_gap": args.max_gap,
            "best_epoch": best_epoch,
            "trained_epochs": args.epochs,
        },
        folder / "model.pt",
    )

    pd.DataFrame(history).to_csv(
        folder / "training_history.csv",
        index=False,
    )

    if window_audit is not None:
        save_json(
            folder / "window_audit.json",
            window_audit,
        )

    runtime = {
        "device": str(device),
        "torch": str(torch.__version__),
        "cuda_runtime": torch.version.cuda,
        "gpu": (
            torch.cuda.get_device_name(device)
            if use_cuda else None
        ),
        "fit_seconds": float(fit_seconds),
        "test_inference_seconds": float(inference_seconds),
        "best_epoch": best_epoch,
        "trained_epochs": args.epochs,
    }

    return (
        validation_scores,
        test_scores,
        indices,
        clean_validation,
        runtime,
    )


def run_one(
    args,
    frames,
    features,
    n_classes,
    seed,
    folder,
):
    arrays = {
        split: frame[features].to_numpy(
            dtype=np.float64
        )
        for split, frame in frames.items()
    }
    targets = {
        split: frame[args.target].to_numpy(
            dtype=np.int64
        )
        for split, frame in frames.items()
    }

    if args.model == "lstm_ae":
        fit_mask = targets["train"] == 0
    else:
        fit_mask = np.ones(
            len(targets["train"]),
            dtype=bool,
        )

    if not fit_mask.any():
        raise ValueError(
            "No eligible observations for preprocessing"
        )

    fitting_data = arrays["train"][fit_mask]
    empty_features = np.isnan(fitting_data).all(axis=0)

    if empty_features.any():
        names = [
            name
            for name, empty in zip(features, empty_features)
            if empty
        ]
        raise ValueError(
            "Features completely missing in eligible training "
            f"data: {names}. Remove or fix these features."
        )

    preprocessing = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    keep_empty_features=True,
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    preprocessing.fit(fitting_data)

    for split, values in arrays.items():
        transformed = preprocessing.transform(values)
        require_finite(
            f"{split}: values after preprocessing",
            transformed,
        )

        with np.errstate(over="ignore", invalid="ignore"):
            transformed = transformed.astype(np.float32)

        require_finite(
            f"{split}: values after float32 conversion",
            transformed,
        )
        arrays[split] = transformed

    joblib.dump(
        preprocessing,
        folder / "preprocessing.joblib",
    )
    save_json(folder / "features.json", features)

    indices = {
        split: np.arange(len(target))
        for split, target in targets.items()
    }
    clean_validation = None

    if args.model in {"mlp", "lstm_ae"}:
        (
            validation_scores,
            test_scores,
            indices,
            clean_validation,
            runtime,
        ) = neural_model(
            args,
            arrays,
            targets,
            n_classes,
            frames,
            seed,
            folder,
            features,
        )

    else:
        if args.model == "logistic":
            model = LogisticRegression(
                max_iter=3000,
                class_weight="balanced",
                random_state=seed,
            )
        else:
            model = RandomForestClassifier(
                n_estimators=args.trees,
                class_weight="balanced_subsample",
                random_state=seed,
                n_jobs=args.jobs,
            )

        started = time.perf_counter()

        model.fit(
            arrays["train"],
            targets["train"],
        )

        fit_seconds = time.perf_counter() - started

        if not np.array_equal(
            model.classes_,
            np.arange(n_classes),
        ):
            raise RuntimeError(
                "Unexpected class order returned by estimator"
            )

        validation_scores = model.predict_proba(
            arrays["validation"]
        )

        started = time.perf_counter()

        test_scores = model.predict_proba(
            arrays["test"]
        )

        runtime = {
            "device": "cpu",
            "fit_seconds": float(fit_seconds),
            "test_inference_seconds": float(
                time.perf_counter() - started
            ),
        }

        joblib.dump(
            model,
            folder / "model.joblib",
        )

    require_finite(
        "Validation scores",
        validation_scores,
    )
    require_finite(
        "Test scores",
        test_scores,
    )

    validation_target = targets["validation"][
        indices["validation"]
    ]
    test_target = targets["test"][
        indices["test"]
    ]

    threshold = None

    if args.model == "lstm_ae":
        if (
            clean_validation is None
            or not clean_validation.any()
        ):
            raise ValueError(
                "No fully normal validation windows "
                "for calibration"
            )

        threshold = float(
            np.quantile(
                validation_scores[clean_validation],
                args.quantile,
                method="higher",
            )
        )

        prediction = (
            test_scores > threshold
        ).astype(np.int64)

        evaluation_score = test_scores
        score_kind = "reconstruction_mse"
        comparison = ">"

    elif n_classes == 2:
        threshold_metric = getattr(
            args,
            "threshold_metric",
            "balanced_accuracy",
        )
        threshold_selector = {
            "f1": select_threshold,
            "balanced_accuracy": select_balanced_accuracy_threshold,
        }.get(threshold_metric)
        if threshold_selector is None:
            raise ValueError(
                f"Unsupported binary threshold metric: {threshold_metric}"
            )
        threshold = threshold_selector(
            validation_target,
            validation_scores[:, 1],
        )

        evaluation_score = test_scores[:, 1]
        prediction = (
            evaluation_score >= threshold
        ).astype(np.int64)

        score_kind = "positive_class_score"
        comparison = ">="

    else:
        prediction = (
            scania_cost_decision(test_scores)
            if args.scania_cost
            else np.argmax(test_scores, axis=1)
        )
        evaluation_score = None
        score_kind = "class_scores"
        comparison = None

    result = metrics(
        test_target,
        prediction,
        evaluation_score,
        n_classes,
        args.scania_cost,
    )
    result.update(
        seed=seed,
        threshold=threshold,
        **runtime,
    )

    save_json(folder / "metrics.json", result)

    save_json(
        folder / "decision.json",
        {
            "threshold": threshold,
            "comparison": comparison,
            "score_kind": score_kind,
            "threshold_selection": (
                "normal validation quantile"
                if args.model == "lstm_ae"
                else (
                    (
                        "maximum validation balanced accuracy"
                        if getattr(
                            args,
                            "threshold_metric",
                            "balanced_accuracy",
                        ) == "balanced_accuracy"
                        else "maximum validation F1"
                    )
                    if n_classes == 2
                    else (
                        "minimum expected SCANIA cost"
                        if args.scania_cost
                        else "argmax class score"
                    )
                )
            ),
            "prediction_rule": (
                "minimum expected cost from class scores"
                if args.scania_cost
                else (
                    "reconstruction score threshold"
                    if args.model == "lstm_ae"
                    else (
                        "positive-class score threshold"
                        if n_classes == 2
                        else "argmax class score"
                    )
                )
            ),
            "evaluation_cost_matrix": (
                SCANIA_COST_MATRIX.tolist()
                if args.scania_cost
                else None
            ),
            "window_label": (
                "last point; no point adjustment"
                if args.model == "lstm_ae"
                else None
            ),
            "quantile": (
                args.quantile
                if args.model == "lstm_ae"
                else None
            ),
            "max_gap": (
                args.max_gap
                if args.model == "lstm_ae"
                else None
            ),
        },
    )

    for split, scores in [
        ("validation", validation_scores),
        ("test", test_scores),
    ]:
        chosen = indices[split]
        source = frames[split].iloc[chosen]

        columns = [args.id]
        if args.group:
            columns.append(args.group)
        if args.time:
            columns.append(args.time)

        output = source[
            list(dict.fromkeys(columns))
        ].copy()

        output["y_true"] = targets[split][chosen]

        if args.model == "lstm_ae":
            output["score"] = scores
            output["y_pred"] = (
                scores > threshold
            ).astype(np.int64)

        else:
            for class_index in range(n_classes):
                output[f"score_{class_index}"] = (
                    scores[:, class_index]
                )

            if n_classes == 2:
                output["y_pred"] = (
                    scores[:, 1] >= threshold
                ).astype(np.int64)
            elif args.scania_cost:
                output["y_pred"] = scania_cost_decision(scores)
            else:
                output["y_pred"] = np.argmax(scores, axis=1)

        output.to_csv(
            folder / f"{split}_predictions.csv",
            index=False,
        )

    majority = int(
        np.bincount(
            targets["train"],
            minlength=n_classes,
        ).argmax()
    )

    baseline_classes = {
        "train_majority": majority,
    }

    if n_classes == 2:
        baseline_classes.update(
            always_normal=0,
            always_positive=1,
        )
    else:
        baseline_classes.update(
            {
                f"always_class_{index}": index
                for index in range(n_classes)
            }
        )

    baselines = {}

    for name, value in baseline_classes.items():
        baseline_prediction = np.full(
            len(test_target),
            value,
            dtype=np.int64,
        )

        baseline_score = (
            np.full(
                len(test_target),
                float(value),
            )
            if n_classes == 2
            else None
        )

        baselines[name] = metrics(
            test_target,
            baseline_prediction,
            baseline_score,
            n_classes,
            args.scania_cost,
        )

    save_json(
        folder / "baselines.json",
        baselines,
    )

    return {
        name: value
        for name, value in result.items()
        if not isinstance(value, (list, dict))
    }


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Reproducible classification and LSTM "
            "reconstruction experiments"
        )
    )

    parser.add_argument("--train", required=True)
    parser.add_argument("--validation", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--features", required=True)

    parser.add_argument("--target", default="target")
    parser.add_argument("--id", default="sample_id")
    parser.add_argument("--group")
    parser.add_argument("--time")

    parser.add_argument(
        "--model",
        required=True,
        choices=[
            "logistic",
            "random_forest",
            "mlp",
            "lstm_ae",
        ],
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[11, 22, 33, 44, 55],
    )

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--latent", type=int, default=16)
    parser.add_argument("--window", type=int, default=30)
    parser.add_argument(
        "--endpoint-only-evaluation",
        action="store_true",
        help=(
            "For LSTM AE, score only each group's final complete window "
            "on validation and test"
        ),
    )
    parser.add_argument(
        "--max-train-windows-per-group",
        type=int,
        help=(
            "Optional cap on fully normal LSTM training windows per group; "
            "selected evenly over each group's timeline"
        ),
    )
    parser.add_argument("--trees", type=int, default=200)
    parser.add_argument("--jobs", type=int, default=1)

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
    )
    parser.add_argument(
        "--quantile",
        type=float,
        default=0.99,
    )
    parser.add_argument(
        "--threshold-metric",
        choices=["balanced_accuracy", "f1"],
        default="balanced_accuracy",
        help=(
            "Binary validation threshold objective; balanced accuracy "
            "is less sensitive to class prevalence than F1"
        ),
    )
    parser.add_argument(
        "--max-gap",
        type=float,
        default=None,
        help=(
            "Maximum time gap allowed inside LSTM windows; "
            "uses the same units as --time"
        ),
    )
    parser.add_argument(
        "--scania-cost",
        action="store_true",
        help=(
            "Use the official five-class SCANIA cost matrix for "
            "decisions and evaluation"
        ),
    )
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--output")

    args = parser.parse_args()

    positive_integers = [
        args.epochs,
        args.batch,
        args.hidden,
        args.latent,
        args.trees,
    ]

    if min(positive_integers) < 1:
        parser.error(
            "epochs, batch, hidden, latent and trees "
            "must be positive integers"
        )

    if args.window < 2:
        parser.error("window must be at least 2")

    if (
        args.max_train_windows_per_group is not None
        and args.max_train_windows_per_group < 1
    ):
        parser.error("--max-train-windows-per-group must be positive")

    if (
        args.model != "lstm_ae"
        and (
            args.endpoint_only_evaluation
            or args.max_train_windows_per_group is not None
        )
    ):
        parser.error(
            "endpoint-only-evaluation and max-train-windows-per-group "
            "are only applicable to lstm_ae"
        )

    if args.jobs == 0:
        parser.error("jobs cannot be 0")

    if (
        not np.isfinite(args.learning_rate)
        or args.learning_rate <= 0
    ):
        parser.error(
            "learning-rate must be finite and positive"
        )

    if (
        not np.isfinite(args.quantile)
        or not 0 < args.quantile < 1
    ):
        parser.error("quantile must be in (0,1)")

    if args.max_gap is not None:
        if (
            not np.isfinite(args.max_gap)
            or args.max_gap <= 0
        ):
            parser.error(
                "max-gap must be finite and positive"
            )
        if args.model != "lstm_ae":
            parser.error(
                "max-gap is only applicable to lstm_ae"
            )

    if not args.seeds:
        parser.error("At least one seed is required")

    if len(args.seeds) != len(set(args.seeds)):
        parser.error("Seeds must be distinct")

    if any(
        seed < 0 or seed >= 2**32
        for seed in args.seeds
    ):
        parser.error(
            "Seeds must satisfy 0 <= seed < 2**32"
        )

    metadata_columns = [
        value
        for value in [
            args.target,
            args.id,
            args.group,
            args.time,
        ]
        if value is not None
    ]
    if any(not value.strip() for value in metadata_columns):
        parser.error("target, id, group and time must be non-empty")
    if len({value.strip().casefold() for value in metadata_columns}) != len(
        metadata_columns
    ):
        parser.error(
            "target, id, group and time must be different columns"
        )

    if args.model == "lstm_ae":
        if not args.group or not args.time:
            parser.error(
                "lstm_ae requires --group and --time"
            )

    if (
        args.device == "cuda"
        and args.model in {"logistic", "random_forest"}
    ):
        parser.error(
            "This estimator is CPU-only"
        )

    if not args.validate_only and not args.output:
        parser.error("--output is required unless --validate-only is set")

    if args.output and not args.validate_only and Path(args.output).exists():
        parser.error(
            "Output path already exists. Use a new directory "
            "to avoid overwriting previous results."
        )

    return args


def write_environment_files(output):
    save_json(
        output / "environment.json",
        environment(),
    )

    try:
        completed = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )

        (output / "pip_freeze.txt").write_text(
            completed.stdout,
            encoding="utf-8",
        )

        if completed.returncode != 0:
            (output / "pip_freeze_error.txt").write_text(
                completed.stderr,
                encoding="utf-8",
            )

    except (OSError, subprocess.TimeoutExpired) as exc:
        (output / "pip_freeze_error.txt").write_text(
            str(exc),
            encoding="utf-8",
        )


def main():
    args = parse_arguments()

    frames, features, n_classes, audit = load_data(args)

    if args.validate_only:
        window_audit = None
        if args.model == "lstm_ae":
            targets = {
                split: frame[args.target].to_numpy(dtype=np.int64)
                for split, frame in frames.items()
            }
            specifications = {}
            for split in frames:
                specifications[split] = build_window_index(
                    frames[split],
                    targets[split],
                    window_index_arguments(args, split),
                )
            if not specifications["train"]["clean"].any():
                raise ValueError("No fully normal training windows")
            if not specifications["validation"]["clean"].any():
                raise ValueError(
                    "No fully normal validation windows for threshold "
                    "calibration"
                )
            window_audit = {
                split: {
                    "all_windows": int(len(specification["starts"])),
                    "fully_normal_windows": int(
                        specification["clean"].sum()
                    ),
                    "fully_normal_windows_before_limit": int(
                        specification[
                            "fully_normal_windows_before_limit"
                        ]
                    ),
                }
                for split, specification in specifications.items()
            }

        print(
            json.dumps(
                {
                    "status": "validated",
                    "features": features,
                    "n_classes": n_classes,
                    "rows": {
                        split: int(len(frame))
                        for split, frame in frames.items()
                    },
                    "window_audit": window_audit,
                },
                ensure_ascii=False,
                allow_nan=False,
            ),
            flush=True,
        )
        return

    if args.scania_cost and n_classes != 5:
        raise ValueError(
            "--scania-cost requires exactly five classes"
        )

    output = Path(args.output)
    output.mkdir(
        parents=True,
        exist_ok=False,
    )

    save_json(
        output / "arguments.json",
        vars(args),
    )
    save_json(
        output / "features.json",
        features,
    )
    save_json(
        output / "data_audit.json",
        audit,
    )

    write_environment_files(output)

    save_json(
        output / "status.json",
        {
            "status": "running",
            "completed_seeds": [],
        },
    )

    runs = []

    try:
        for seed in args.seeds:
            folder = output / f"seed_{seed}"
            folder.mkdir()

            result = run_one(
                args,
                frames,
                features,
                n_classes,
                seed,
                folder,
            )
            runs.append(result)

            # Промежуточные результаты не теряются при ошибке
            # на одном из следующих запусков.
            pd.DataFrame(runs).to_csv(
                output / "runs.csv",
                index=False,
            )

            save_json(
                output / "status.json",
                {
                    "status": "running",
                    "completed_seeds": [
                        item["seed"] for item in runs
                    ],
                },
            )

            print(
                json.dumps(
                    result,
                    ensure_ascii=False,
                    allow_nan=False,
                    default=json_default,
                ),
                flush=True,
            )

        table = pd.DataFrame(runs)
        summary = []

        for column in table.select_dtypes(
            include="number"
        ).columns:
            if column in {
                "seed",
                "best_epoch",
                "trained_epochs",
            }:
                continue

            values = table[column].dropna()

            summary.append(
                {
                    "metric": column,
                    "n_runs": int(len(values)),
                    "mean": (
                        float(values.mean())
                        if len(values) else None
                    ),
                    "sample_std": (
                        float(values.std(ddof=1))
                        if len(values) > 1
                        else None
                    ),
                }
            )

        pd.DataFrame(summary).to_csv(
            output / "summary.csv",
            index=False,
        )
        save_json(
            output / "summary.json",
            {
                "metrics": summary,
                "interpretation_note": (
                    "Variation across seeds on this fixed split is not "
                    "a generalization confidence interval."
                ),
            },
        )

        save_json(
            output / "status.json",
            {
                "status": "completed",
                "completed_seeds": [
                    item["seed"] for item in runs
                ],
            },
        )

    except BaseException as exc:
        status = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        try:
            save_json(
                output / "status.json",
                {
                    "status": status,
                    "completed_seeds": [
                        item["seed"] for item in runs
                    ],
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
        except Exception as status_error:
            if hasattr(exc, "add_note"):
                exc.add_note(
                    "Also failed to write run status: "
                    f"{type(status_error).__name__}: {status_error}"
                )
        raise

    print(
        "Results:",
        output.resolve(),
        flush=True,
    )


if __name__ == "__main__":
    main()