from __future__ import annotations

import argparse
import hashlib
import json
from decimal import Decimal, InvalidOperation
from pathlib import Path

import pandas as pd


SOURCE_LABEL_COLUMNS = {
    "label",
    "target",
    "class",
    "class_label",
    "attack",
    "attack_type",
    "attack_name",
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--chunksize", type=int, default=50000)
    parser.add_argument(
        "--profiling-train-fraction",
        type=float,
        default=0.0,
        help=(
            "Fraction of profiling captures to move from validation "
            "into train; captures remain intact and split deterministically"
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.chunksize < 1:
        parser.error("--chunksize must be a positive integer")
    if (
        not 0 <= args.seed < 2**32
        or not 0 <= args.profiling_train_fraction < 1
    ):
        parser.error(
            "seed must be in [0, 2**32) and "
            "profiling-train-fraction must be in [0, 1)"
        )

    root = Path(args.root).resolve()
    output = Path(args.output)

    features = json.loads(
        Path(args.features).read_text(encoding="utf-8")
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
        raise ValueError("Invalid feature list")

    forbidden = {
        "target", "label", "sample_id", "sequence_id",
        "source_id", "filename", "attack", "attack_name",
    }
    if any(name.strip().lower() in forbidden for name in features):
        raise ValueError("Metadata or label included in features")

    manifest = pd.read_csv(
        args.manifest,
        dtype={
            "split": str,
            "source_partition": str,
            "source_id": str,
            "path": str,
            "target": str,
            "source_label": str,
        },
    )

    required = {
        "split",
        "source_partition",
        "source_id",
        "path",
        "target",
    }
    if not required.issubset(manifest.columns):
        raise ValueError(f"Manifest requires: {sorted(required)}")

    if manifest[list(required)].isna().any().any():
        raise ValueError("Missing manifest values")
    for column in ("source_partition", "source_id", "path"):
        values = manifest[column].astype(str)
        if (
            values.str.strip().eq("").any()
            or values.str.strip().ne(values).any()
        ):
            raise ValueError(
                f"Manifest {column} values must be non-empty and have "
                "no surrounding whitespace"
            )
    if "source_label" in manifest.columns:
        if manifest["source_label"].isna().any() or (
            manifest["source_label"].astype(str).str.strip().eq("").any()
        ):
            raise ValueError("Manifest source_label values must be non-empty")

    if set(manifest["split"]) != {"train", "validation", "test"}:
        raise ValueError("Provide train, validation and test")
    valid_partition_assignment = (
        (
            manifest["split"].eq("train")
            & manifest["source_partition"].isin(["train", "profiling"])
        )
        | (
            manifest["split"].eq("validation")
            & manifest["source_partition"].isin(["train", "profiling"])
        )
        | (
            manifest["split"].eq("test")
            & manifest["source_partition"].eq("test")
        )
    )
    if not valid_partition_assignment.all():
        raise ValueError(
            "Validation sources must come from the official training "
            "partition or a separate profiling partition; profiling "
            "captures may be assigned to train or validation but never "
            "split across them, and test sources must remain in the official "
            "test partition"
        )

    parsed_targets = []
    for value in manifest["target"]:
        try:
            target = Decimal(str(value).strip())
        except (InvalidOperation, TypeError, ValueError) as exc:
            raise ValueError("Manifest targets must be binary integers") from exc
        if not target.is_finite() or target not in (Decimal(0), Decimal(1)):
            raise ValueError("Only binary targets 0 and 1 are supported")
        parsed_targets.append(int(target))
    manifest["target"] = parsed_targets

    profiling_validation = manifest[
        manifest["source_partition"].eq("profiling")
        & manifest["split"].eq("validation")
    ]["source_id"].drop_duplicates().tolist()
    profiling_train_ids = []
    if args.profiling_train_fraction:
        if len(profiling_validation) < 2:
            raise ValueError(
                "At least two profiling captures are required to split "
                "profiling sources between train and validation"
            )
        ordered_ids = sorted(
            profiling_validation,
            key=lambda source_id: hashlib.sha256(
                f"{args.seed}:{source_id}".encode("utf-8")
            ).hexdigest(),
        )
        train_count = round(
            len(ordered_ids) * args.profiling_train_fraction
        )
        train_count = min(max(train_count, 1), len(ordered_ids) - 1)
        profiling_train_ids = ordered_ids[:train_count]
        manifest.loc[
            manifest["source_id"].isin(profiling_train_ids),
            "split",
        ] = "train"

    if manifest.groupby("source_id")["split"].nunique().max() > 1:
        raise ValueError("A source capture occurs in different splits")

    paths = []
    for relative in manifest["path"]:
        path = (root / relative).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Path escapes dataset root")
        if not path.is_file():
            raise FileNotFoundError(path)
        paths.append(path)

    if len(paths) != len(set(paths)):
        raise ValueError("The same CSV is listed more than once")

    source_label_columns = {}
    for path in paths:
        columns = pd.read_csv(path, nrows=0).columns
        missing_features = sorted(set(features) - set(columns))
        if missing_features:
            raise ValueError(
                f"{path.name}: missing feature columns: {missing_features}"
            )
        source_label_columns[path] = [
            column
            for column in columns
            if str(column).strip().lower() in SOURCE_LABEL_COLUMNS
        ]

    output.mkdir(parents=True, exist_ok=False)

    written = {name: 0 for name in ["train", "validation", "test"]}
    counts = {
        name: {0: 0, 1: 0}
        for name in written
    }
    records = []
    source_hash_paths = {}

    for row, path in zip(manifest.itertuples(index=False), paths):
        split = str(row.split)
        source_id = str(row.source_id)
        source_partition = str(row.source_partition)
        target = int(str(row.target))
        digest = sha256(path)
        if digest in source_hash_paths:
            previous_path, previous_split = source_hash_paths[digest]
            raise ValueError(
                "Duplicate source-file content: "
                f"{previous_path} ({previous_split}) and "
                f"{path.relative_to(root).as_posix()} ({split})"
            )
        source_hash_paths[digest] = (
            path.relative_to(root).as_posix(),
            split,
        )

        offset = 0
        relative = path.relative_to(root).as_posix()
        destination = output / f"{split}.csv"
        label_columns = source_label_columns[path]
        mapping_label_column = next(
            (
                column
                for preferred in (
                    "label",
                    "target",
                    "class_label",
                    "class",
                    "attack",
                    "attack_type",
                    "attack_name",
                )
                for column in label_columns
                if str(column).strip().lower() == preferred
            ),
            None,
        )
        observed_source_labels = {
            column: set() for column in label_columns
        }
        columns_to_read = list(dict.fromkeys(features + label_columns))

        for chunk in pd.read_csv(
            path,
            usecols=columns_to_read,
            dtype={column: str for column in label_columns},
            chunksize=args.chunksize,
        ):
            for column in label_columns:
                values = chunk[column]
                if values.isna().any() or values.astype(str).str.strip().eq(
                    ""
                ).any():
                    raise ValueError(
                        f"{relative}: missing values in source label "
                        f"column {column!r}"
                    )
                observed_source_labels[column].update(
                    values.astype(str).str.strip().unique()
                )
                if len(observed_source_labels[column]) > 1:
                    raise ValueError(
                        f"{relative}: mixed labels in source file "
                        f"column {column!r}"
                    )

            chunk = chunk[features].copy()

            for column in features:
                chunk[column] = pd.to_numeric(
                    chunk[column], errors="raise"
                )

            size = len(chunk)
            chunk.insert(
                0, "sample_id",
                [f"{relative}::{i}" for i in range(offset, offset + size)],
            )
            chunk.insert(1, "sequence_id", source_id)
            chunk["target"] = target

            chunk.to_csv(
                destination,
                mode="a",
                header=written[split] == 0,
                index=False,
            )

            written[split] += size
            counts[split][target] += size
            offset += size

        if not offset:
            raise ValueError(f"{relative}: source file contains no rows")

        if label_columns:
            mapping_label = next(
                iter(observed_source_labels[mapping_label_column])
            )
            if "source_label" in manifest.columns:
                source_label = str(row.source_label).strip()
                if mapping_label != source_label:
                    raise ValueError(
                        f"{relative}: source label {mapping_label!r} "
                        "does not match manifest source_label "
                        f"{source_label!r}"
                    )
            else:
                try:
                    numeric_label = Decimal(mapping_label)
                except InvalidOperation as exc:
                    raise ValueError(
                        f"{relative}: add a source_label column to the "
                        "manifest to document the mapping from this "
                        "source label to target"
                    ) from exc
                if (
                    not numeric_label.is_finite()
                    or numeric_label not in (Decimal(0), Decimal(1))
                    or numeric_label != target
                ):
                    raise ValueError(
                        f"{relative}: numeric source label "
                        f"{mapping_label!r} does not match manifest "
                        f"target {row.target}"
                    )

        records.append({
            "path": relative,
            "source_id": source_id,
            "source_partition": source_partition,
            "split": split,
            "target": target,
            "source_label": (
                {
                    column: next(iter(values))
                    for column, values in observed_source_labels.items()
                }
                if label_columns else None
            ),
            "rows": offset,
            "sha256": digest,
        })

    for split in written:
        if not written[split] or min(counts[split].values()) == 0:
            raise ValueError(
                f"{split}: both normal and attack observations required"
            )

    (output / "features.json").write_text(
        json.dumps(features, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output / "preparation.json").write_text(
        json.dumps({
            "dataset": "CICIoMT2024",
            "task": "binary classification",
            "features": features,
            "class_counts": counts,
            "files": records,
            "label_note": (
                "The manifest target is the configured binary mapping; "
                "verify it against authoritative dataset documentation. "
                "Recognized source label columns are checked for a single "
                "value per file; non-numeric source labels require an "
                "explicit manifest source_label value."
            ),
            "split_provenance_note": (
                "Profiling captures can be split by source_id between "
                "train and validation using profiling_train_fraction and "
                "seed. This preserves capture boundaries and exposes both "
                "splits to the profiling domain, but is not an official "
                "attack-dataset validation split. Official test sources "
                "remain in the test split."
            ),
            "profiling_split": {
                "train_fraction": args.profiling_train_fraction,
                "seed": args.seed,
                "captures_moved_to_train": len(profiling_train_ids),
                "captures_held_out_for_validation": (
                    len(profiling_validation) - len(profiling_train_ids)
                ),
            },
            "source_independence_note": (
                "Unique sample IDs and source files do not establish "
                "statistical independence; source provenance and split "
                "design must be justified separately."
            ),
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(counts, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()