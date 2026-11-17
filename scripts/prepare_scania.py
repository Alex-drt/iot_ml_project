from __future__ import annotations

import argparse
import hashlib
import json
from decimal import Decimal, InvalidOperation
from pathlib import Path

import numpy as np
import pandas as pd


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_table(path, required):
    frame = pd.read_csv(
        path,
        dtype={
            "vehicle_id": str,
            "class_label": str,
            "in_study_repair": str,
        },
    )

    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(
            f"{path.name}: missing {sorted(missing)}. "
            "Do not join tables by row order; verify vehicle_id."
        )
    if (
        frame["vehicle_id"].isna().any()
        or frame["vehicle_id"].str.strip().eq("").any()
        or frame["vehicle_id"].str.strip().ne(frame["vehicle_id"]).any()
    ):
        raise ValueError(
            f"{path.name}: vehicle_id must be non-empty and have no "
            "surrounding whitespace"
        )
    return frame


def validate_readouts(frame):
    frame["time_step"] = pd.to_numeric(
        frame["time_step"], errors="raise"
    )
    if not np.isfinite(frame["time_step"].to_numpy(float)).all():
        raise ValueError("Invalid time_step")
    if frame.duplicated(["vehicle_id", "time_step"]).any():
        raise ValueError("Duplicate vehicle_id/time_step")
    return frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--training-sampling",
        choices=["one_per_vehicle", "all_eligible_readouts"],
        default="all_eligible_readouts",
    )
    parser.add_argument(
        "--write-anomaly-sequences",
        action="store_true",
        help=(
            "Write full eligible train and labeled validation/test "
            "readout sequences for endpoint-evaluated LSTM anomaly detection"
        ),
    )
    args = parser.parse_args()

    if not 0 <= args.seed < 2**32:
        parser.error("seed must satisfy 0 <= seed < 2**32")
    if (
        args.write_anomaly_sequences
        and args.training_sampling != "all_eligible_readouts"
    ):
        parser.error(
            "anomaly sequences require all_eligible_readouts training data"
        )

    root = Path(args.root)
    output = Path(args.output)
    rng = np.random.default_rng(args.seed)

    names = [
        "train_operational_readouts.csv",
        "train_tte.csv",
        "validation_operational_readouts.csv",
        "validation_labels.csv",
        "test_operational_readouts.csv",
        "test_labels.csv",
    ]
    paths = {name: root / name for name in names}
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)

    train = validate_readouts(read_table(
        paths["train_operational_readouts.csv"],
        ["vehicle_id", "time_step"],
    ))

    features = [
        name for name in train.columns
        if name not in {"vehicle_id", "time_step"}
    ]

    forbidden = {
        "target", "class_label", "in_study_repair",
        "length_of_study_time_step", "rul", "time_to_event",
    }
    if (
        not features
        or len({name.casefold() for name in features}) != len(features)
        or any(
            not name.strip()
            or name != name.strip()
            or name.lower() in forbidden
            for name in features
        )
    ):
        raise ValueError("Invalid operational feature schema")

    tte = read_table(
        paths["train_tte.csv"],
        ["vehicle_id", "length_of_study_time_step", "in_study_repair"],
    )
    if tte["vehicle_id"].duplicated().any():
        raise ValueError("Duplicate vehicle in train_tte")

    tte["length_of_study_time_step"] = pd.to_numeric(
        tte["length_of_study_time_step"], errors="raise"
    )
    event_flags = []
    for value in tte["in_study_repair"]:
        try:
            event_flag = Decimal(str(value).strip())
        except (InvalidOperation, TypeError, ValueError) as exc:
            raise ValueError(
                "in_study_repair must contain binary integers"
            ) from exc
        if (
            not event_flag.is_finite()
            or event_flag not in (Decimal(0), Decimal(1))
        ):
            raise ValueError("Invalid event information")
        event_flags.append(int(event_flag))
    tte["in_study_repair"] = event_flags

    if (
        not np.isfinite(
            tte["length_of_study_time_step"].to_numpy(float)
        ).all()
    ):
        raise ValueError("Invalid event information")

    train = train.merge(
        tte,
        on="vehicle_id",
        how="left",
        validate="many_to_one",
        indicator=True,
    )
    if not train["_merge"].eq("both").all():
        raise ValueError("Readouts without event/follow-up information")
    train = train.drop(columns="_merge")

    remaining = (
        train["length_of_study_time_step"] - train["time_step"]
    )
    if not np.isfinite(remaining.to_numpy(float)).all():
        raise ValueError("Invalid remaining time after event/follow-up join")
    event = train["in_study_repair"].eq(1)

    eligible = (
        (event & remaining.ge(0))
        | (~event & remaining.gt(48))
    )

    train = train.loc[eligible].copy()
    r = remaining.loc[eligible]

    train["target"] = np.select(
        [r.gt(48), r.gt(24), r.gt(12), r.gt(6)],
        [0, 1, 2, 3],
        default=4,
    ).astype(int)

    train = train.sort_values(
        ["vehicle_id", "time_step"], kind="stable"
    ).reset_index(drop=True)

    if args.training_sampling == "one_per_vehicle":
        chosen = []
        for positions in train.groupby(
            "vehicle_id", sort=True
        ).indices.values():
            chosen.append(int(rng.choice(positions)))
        train = train.iloc[chosen].copy()

    prepared = {"train": train}
    sequence_prepared = {}
    if args.write_anomaly_sequences:
        train = train.copy()
        train["anomaly_target"] = train["target"].gt(0).astype(int)
        sequence_prepared["train"] = train.copy()

    for split in ["validation", "test"]:
        readouts = validate_readouts(read_table(
            paths[f"{split}_operational_readouts.csv"],
            ["vehicle_id", "time_step"],
        ))

        if set(readouts.columns) != set(
            ["vehicle_id", "time_step"] + features
        ):
            raise ValueError(f"{split}: operational schema differs")

        labels = read_table(
            paths[f"{split}_labels.csv"],
            ["vehicle_id", "class_label"],
        )
        if labels["vehicle_id"].duplicated().any():
            raise ValueError(f"{split}: duplicate labels")

        parsed_labels = []
        for value in labels["class_label"]:
            try:
                label = Decimal(str(value).strip())
            except (InvalidOperation, TypeError, ValueError) as exc:
                raise ValueError(
                    f"{split}: class labels must be integers 0 through 4"
                ) from exc
            if (
                not label.is_finite()
                or label != label.to_integral_value()
                or label < 0
                or label > 4
            ):
                raise ValueError(f"{split}: invalid class labels")
            parsed_labels.append(int(label))
        labels["class_label"] = parsed_labels

        if args.write_anomaly_sequences:
            sequence_frame = readouts.merge(
                labels[["vehicle_id", "class_label"]],
                on="vehicle_id",
                how="inner",
                validate="many_to_one",
            ).copy()
            sequence_frame["target"] = sequence_frame[
                "class_label"
            ].astype(int)
            sequence_frame["anomaly_target"] = (
                sequence_frame["target"].gt(0).astype(int)
            )
            sequence_prepared[split] = sequence_frame

        last = (
            readouts.sort_values(
                ["vehicle_id", "time_step"], kind="stable"
            )
            .groupby("vehicle_id", sort=False)
            .tail(1)
        )

        if set(last["vehicle_id"]) != set(labels["vehicle_id"]):
            raise ValueError(f"{split}: vehicle IDs do not match labels")

        frame = last.merge(
            labels,
            on="vehicle_id",
            how="inner",
            validate="one_to_one",
        )
        frame["target"] = frame["class_label"].astype(int)
        prepared[split] = frame

    for a, b in [
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    ]:
        if set(prepared[a]["vehicle_id"]) & set(prepared[b]["vehicle_id"]):
            raise ValueError(f"Vehicle overlap: {a}/{b}")

    if set(prepared["train"]["target"]) != set(range(5)):
        raise ValueError(
            "Prepared training data does not contain all five classes"
        )

    output.mkdir(parents=True, exist_ok=False)
    counts = {}

    for split, frame in prepared.items():
        for name in features:
            frame[name] = pd.to_numeric(frame[name], errors="raise")

        frame = frame[
            ["vehicle_id", "time_step"] + features + ["target"]
        ].copy()

        frame.insert(
            0, "sample_id",
            [
                f"{split}::{vehicle}::{step}"
                for vehicle, step in zip(
                    frame["vehicle_id"], frame["time_step"]
                )
            ],
        )

        frame.to_csv(output / f"{split}.csv", index=False)
        counts[split] = {
            str(k): int(v)
            for k, v in frame["target"].value_counts().items()
        }

    if args.write_anomaly_sequences:
        sequence_counts = {}
        for split, frame in sequence_prepared.items():
            for name in features:
                frame[name] = pd.to_numeric(
                    frame[name],
                    errors="raise",
                )
            frame = frame[
                [
                    "vehicle_id",
                    "time_step",
                    *features,
                    "target",
                    "anomaly_target",
                ]
            ].copy()
            frame.insert(
                0,
                "sample_id",
                [
                    f"{split}::{vehicle}::{step}"
                    for vehicle, step in zip(
                        frame["vehicle_id"],
                        frame["time_step"],
                    )
                ],
            )
            frame.to_csv(
                output / f"{split}_anomaly_sequences.csv",
                index=False,
            )
            sequence_counts[split] = {
                "rows": int(len(frame)),
                "vehicles": int(frame["vehicle_id"].nunique()),
                "anomaly_classes": {
                    str(label): int(count)
                    for label, count in frame[
                        "anomaly_target"
                    ].value_counts().items()
                },
            }

    (output / "features.json").write_text(
        json.dumps(features, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    (output / "preparation.json").write_text(
        json.dumps({
            "dataset": "SCANIA Component X",
            "task": "five-class classification",
            "seed": args.seed,
            "training_sampling": args.training_sampling,
            "validation_test_sampling": "last available readout",
            "anomaly_sequence_files": (
                {
                    split: f"{split}_anomaly_sequences.csv"
                    for split in sequence_prepared
                }
                if args.write_anomaly_sequences
                else None
            ),
            "anomaly_sequence_counts": (
                sequence_counts
                if args.write_anomaly_sequences
                else None
            ),
            "time_interpretation": (
                "remaining = length_of_study_time_step - time_step; "
                "in_study_repair=1 is treated as event time and "
                "in_study_repair=0 as end of censored follow-up"
            ),
            "training_eligibility": (
                "event observations with remaining >= 0; censored "
                "observations only when remaining > 48"
            ),
            "class_boundaries": {
                "0": "remaining > 48",
                "1": "24 < remaining <= 48",
                "2": "12 < remaining <= 24",
                "3": "6 < remaining <= 12",
                "4": "remaining <= 6",
            },
            "label_provenance": [
                (
                    "datasets/archives/scania_component_x/"
                    "SCANIA_Component_X_dataset/2024-34-2/2024-34-2/"
                    "documentation/2024_IDA_challenge_v2.pdf"
                ),
                (
                    "datasets/archives/scania_component_x/"
                    "SCANIA_Component_X_dataset/scientific_data.pdf"
                ),
            ],
            "label_semantics": (
                "Each eligible training readout is labeled by the time "
                "remaining until the first repair event. Censored "
                "readouts are class 0 only when follow-up exceeds 48; "
                "readouts after a repair and censored readouts with "
                "shorter follow-up are excluded. Validation/test labels "
                "are supplied for the selected last readout using the "
                "same time windows."
            ),
            "class_counts": counts,
            "features": features,
            "input_sha256": {
                name: sha256(path) for name, path in paths.items()
            },
            "limitations": [
                "No categorical vehicle specifications",
                "No survival analysis",
                "Sequence anomaly targets for validation/test use the "
                "supplied vehicle-level class; only the last available "
                "window should be evaluated",
                "Validation/test labels are the supplied per-vehicle classes "
                "at the selected last available readout; no separate "
                "event-time table is available for deriving their labels",
                "Training eligibility introduces selection restrictions",
            ],
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(counts, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()