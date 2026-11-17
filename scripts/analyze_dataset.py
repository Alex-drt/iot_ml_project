from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


SPLITS = ("train", "validation", "test")


def _read_features(path: Path) -> list[str]:
    features = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(features, list)
        or not features
        or any(
            not isinstance(name, str)
            or not name.strip()
            or name != name.strip()
            for name in features
        )
        or len({name.casefold() for name in features}) != len(features)
    ):
        raise ValueError(
            "features.json must contain a non-empty list of unique "
            "column names"
        )
    return features


def _validate_inputs(
    dataset: Path,
    features: list[str],
    target: str,
) -> dict[str, Path]:
    paths = {split: dataset / f"{split}.csv" for split in SPLITS}
    for split, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"{split} CSV not found: {path}")
        columns = pd.read_csv(path, nrows=0).columns
        missing = sorted(set(features + [target]) - set(columns))
        if missing:
            raise ValueError(f"{split}: missing columns: {missing}")
    if target in features:
        raise ValueError("The target column cannot also be a feature")
    return paths


def _analyze_split(
    path: Path,
    features: list[str],
    target: str,
    sample_size: int,
    chunksize: int,
    rng: np.random.Generator,
) -> tuple[dict, pd.DataFrame]:
    row_count = 0
    missing_counts = Counter()
    infinite_counts = Counter()
    target_counts = Counter()
    sample = pd.DataFrame(columns=features + [target])

    for chunk in pd.read_csv(
        path,
        usecols=features + [target],
        chunksize=chunksize,
    ):
        if chunk[target].isna().any():
            raise ValueError(f"{path}: target contains missing values")

        for feature in features:
            chunk[feature] = pd.to_numeric(chunk[feature], errors="raise")

        raw_features = chunk[features]
        missing_counts.update(
            raw_features.isna().sum().astype(int).to_dict()
        )
        infinite = np.isinf(raw_features.to_numpy(dtype=np.float64))
        infinite_counts.update(
            dict(zip(features, infinite.sum(axis=0).astype(int)))
        )
        chunk.loc[:, features] = raw_features.replace(
            [np.inf, -np.inf],
            np.nan,
        )
        target_counts.update(
            {
                str(value): int(count)
                for value, count in chunk[target].value_counts().items()
            }
        )

        previous_count = row_count
        row_count += len(chunk)
        new_sample_size = min(sample_size, row_count)
        from_chunk = int(
            rng.hypergeometric(
                len(chunk),
                previous_count,
                new_sample_size,
            )
        )
        from_previous = new_sample_size - from_chunk
        selected = []
        if from_previous:
            selected.append(
                sample.sample(
                    n=from_previous,
                    random_state=int(rng.integers(0, 2**32)),
                )
            )
        if from_chunk:
            selected.append(
                chunk.sample(
                    n=from_chunk,
                    random_state=int(rng.integers(0, 2**32)),
                )
            )
        sample = pd.concat(selected, ignore_index=True)

    if row_count == 0:
        raise ValueError(f"{path}: CSV contains no data rows")

    feature_statistics = {}
    for feature in features:
        values = sample[feature].dropna()
        stats = values.describe()
        feature_statistics[feature] = {
            key: (
                float(stats[key])
                if key in stats and np.isfinite(stats[key])
                else None
            )
            for key in ("mean", "std", "min", "25%", "50%", "75%", "max")
        }

    missing_by_feature = {
        feature: int(missing_counts[feature])
        for feature in features
    }
    infinite_by_feature = {
        feature: int(infinite_counts[feature])
        for feature in features
    }
    return (
        {
            "rows": row_count,
            "target_distribution": dict(sorted(target_counts.items())),
            "target_distribution_percent": {
                label: 100 * count / row_count
                for label, count in sorted(target_counts.items())
            },
            "missing_values_by_feature": missing_by_feature,
            "missing_feature_values": sum(missing_by_feature.values()),
            "infinite_values_by_feature": infinite_by_feature,
            "infinite_feature_values": sum(infinite_by_feature.values()),
            "sampled_rows": int(len(sample)),
            "feature_statistics_on_sample": feature_statistics,
        },
        sample,
    )


def _create_plots(
    output: Path,
    features: list[str],
    split_results: dict[str, dict],
    samples: dict[str, pd.DataFrame],
) -> list[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "matplotlib is required to generate exploratory plots. "
            "Install the project requirements first."
        ) from exc

    created = []

    classes = sorted(
        {
            label
            for result in split_results.values()
            for label in result["target_distribution"]
        }
    )
    figure, axis = plt.subplots(figsize=(max(8, len(classes) * 1.2), 5))
    positions = np.arange(len(classes))
    width = 0.8 / len(SPLITS)
    for index, split in enumerate(SPLITS):
        counts = split_results[split]["target_distribution"]
        rows = split_results[split]["rows"]
        axis.bar(
            positions + (index - (len(SPLITS) - 1) / 2) * width,
            [100 * counts.get(label, 0) / rows for label in classes],
            width,
            label=split,
        )
    axis.set_xticks(positions, classes)
    axis.set_xlabel("Класс целевой переменной")
    axis.set_ylabel("Доля строк, %")
    axis.set_title("Доли классов по выборкам")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output / "class_distribution.png", dpi=150)
    plt.close(figure)
    created.append("class_distribution.png")

    total_rows = sum(result["rows"] for result in split_results.values())
    missing_totals = {
        feature: sum(
            result["missing_values_by_feature"][feature]
            for result in split_results.values()
        )
        for feature in features
    }
    top_missing = sorted(
        features,
        key=lambda feature: missing_totals[feature],
        reverse=True,
    )[:20]
    figure, axis = plt.subplots(figsize=(max(9, len(top_missing) * 0.55), 5))
    axis.bar(
        top_missing,
        [
            100 * missing_totals[feature] / total_rows
            for feature in top_missing
        ],
    )
    axis.set_ylabel("Пропуски, % от всех строк")
    axis.set_title("Пропущенные значения по признакам (все выборки)")
    axis.tick_params(axis="x", labelrotation=60)
    figure.tight_layout()
    figure.savefig(output / "missing_values.png", dpi=150)
    plt.close(figure)
    created.append("missing_values.png")

    train_sample = samples["train"]
    variances = train_sample[features].var().fillna(0).sort_values()
    selected_features = list(variances.tail(min(6, len(features))).index)
    rows = (len(selected_features) + 1) // 2
    figure, axes = plt.subplots(
        rows,
        2,
        figsize=(12, 3.5 * rows),
        squeeze=False,
    )
    for axis, feature in zip(axes.flat, selected_features):
        for split in SPLITS:
            values = samples[split][feature].dropna()
            if not values.empty:
                axis.hist(
                    values,
                    bins=30,
                    density=True,
                    alpha=0.45,
                    label=split,
                )
        axis.set_title(feature)
        axis.set_ylabel("Плотность")
        axis.legend()
    for axis in list(axes.flat)[len(selected_features):]:
        axis.set_visible(False)
    figure.suptitle("Распределения признаков (случайная выборка)")
    figure.tight_layout()
    figure.savefig(output / "feature_distributions.png", dpi=150)
    plt.close(figure)
    created.append("feature_distributions.png")

    correlation_features = list(
        variances.tail(min(15, len(features))).index
    )
    correlation = train_sample[correlation_features].corr()
    figure, axis = plt.subplots(
        figsize=(max(7, len(correlation_features) * 0.55), 6)
    )
    image = axis.imshow(
        np.ma.masked_invalid(correlation.to_numpy()),
        vmin=-1,
        vmax=1,
        cmap="coolwarm",
    )
    axis.set_xticks(
        np.arange(len(correlation_features)),
        correlation_features,
        rotation=70,
        ha="right",
    )
    axis.set_yticks(np.arange(len(correlation_features)), correlation_features)
    axis.set_title("Корреляции признаков (только train, sample)")
    figure.colorbar(image, ax=axis, label="Корреляция Пирсона")
    figure.tight_layout()
    figure.savefig(output / "train_feature_correlation.png", dpi=150)
    plt.close(figure)
    created.append("train_feature_correlation.png")
    return created


def analyze_dataset(
    dataset: Path,
    output: Path,
    *,
    target: str = "target",
    sample_size: int = 10000,
    chunksize: int = 50000,
) -> dict:
    dataset = Path(dataset).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    if sample_size < 1 or chunksize < 1:
        raise ValueError("sample_size and chunksize must be positive")
    if output.exists():
        raise FileExistsError(f"Output directory already exists: {output}")

    features_path = dataset / "features.json"
    if not features_path.is_file():
        raise FileNotFoundError(f"features.json not found: {features_path}")
    features = _read_features(features_path)
    paths = _validate_inputs(dataset, features, target)

    rng = np.random.default_rng(42)
    split_results = {}
    samples = {}
    for split, path in paths.items():
        split_results[split], samples[split] = _analyze_split(
            path,
            features,
            target,
            sample_size,
            chunksize,
            rng,
        )

    output.mkdir(parents=True, exist_ok=False)
    plots = _create_plots(output, features, split_results, samples)
    report = {
        "dataset": str(dataset),
        "target": target,
        "features": features,
        "sample_size_limit": sample_size,
        "random_seed": 42,
        "plots": plots,
        "splits": split_results,
        "interpretation_note": (
            "Counts and missing-value totals use all rows. Feature "
            "statistics and histograms use a uniform random sample; "
            "the correlation plot uses train only. This exploratory "
            "report is not a model evaluation."
        ),
    }
    (output / "analysis.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return report


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Exploratory analysis and plots for prepared CSV splits"
    )
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--target", default="target")
    parser.add_argument("--sample-size", type=int, default=10000)
    parser.add_argument("--chunksize", type=int, default=50000)
    return parser.parse_args()


def main():
    args = parse_arguments()
    report = analyze_dataset(
        args.dataset,
        args.output,
        target=args.target,
        sample_size=args.sample_size,
        chunksize=args.chunksize,
    )
    print(
        json.dumps(
            {
                "status": "completed",
                "output": str(Path(args.output).expanduser().resolve()),
                "rows": {
                    split: result["rows"]
                    for split, result in report["splits"].items()
                },
                "plots": report["plots"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
