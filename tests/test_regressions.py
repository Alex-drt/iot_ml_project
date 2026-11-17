from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_PATH = ROOT / "iot_experiment.py.py"
SPEC = importlib.util.spec_from_file_location(
    "iot_experiment",
    EXPERIMENT_PATH,
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(
        f"Unable to load experiment module from {EXPERIMENT_PATH}"
    )
EXPERIMENT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = EXPERIMENT
SPEC.loader.exec_module(EXPERIMENT)


def fixture_splits(root, *, groups=False, labels=None, all_missing=False):
    root = Path(root)
    feature = "signal"
    ids_by_split = {
        "train": ["tr0", "tr1", "tr2", "tr3"],
        "validation": ["va0", "va1", "va2", "va3"],
        "test": ["te0", "te1", "te2", "te3"],
    }
    values_by_split = {
        "train": [0.0, 1.0, 0.1, 0.9],
        "validation": [0.05, 0.95, 0.2, 0.8],
        "test": [0.02, 0.98, 0.25, 0.75],
    }
    targets_by_split = labels or {
        "train": [0, 1, 0, 1],
        "validation": [0, 1, 0, 1],
        "test": [0, 1, 0, 1],
    }
    for split in ids_by_split:
        values = values_by_split[split]
        if all_missing and split == "train":
            values = [np.nan] * len(values)
        frame = pd.DataFrame(
            {
                "sample_id": ids_by_split[split],
                feature: values,
                "target": targets_by_split[split],
            }
        )
        if groups:
            frame["sequence_id"] = [
                f"{split}_group_{index}" for index in range(len(frame))
            ]
        frame.to_csv(root / f"{split}.csv", index=False)
    (root / "features.json").write_text(
        json.dumps([feature]),
        encoding="utf-8",
    )
    return Namespace(
        train=str(root / "train.csv"),
        validation=str(root / "validation.csv"),
        test=str(root / "test.csv"),
        features=str(root / "features.json"),
        target="target",
        id="sample_id",
        group="sequence_id" if groups else None,
        time=None,
        model="logistic",
        scania_cost=False,
        device="cpu",
        seeds=[11],
        epochs=1,
        batch=2,
        hidden=4,
        latent=2,
        window=2,
        trees=2,
        jobs=1,
        learning_rate=0.01,
        quantile=0.9,
        max_gap=None,
        output=str(root / "output"),
    )


def run_experiment_cli(command):
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise AssertionError(
            f"Experiment exited with {result.returncode}.\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    return result


def write_ciciomt_fixture(root, *, mixed=False, duplicate_content=False):
    root = Path(root)
    manifest_rows = []
    for split_index, split in enumerate(("train", "validation", "test")):
        for label in (0, 1):
            relative = f"{split}_{label}.csv"
            source_labels = [label, label]
            if mixed and split == "train" and label == 0:
                source_labels = [0, 1]
            data = pd.DataFrame(
                {
                    "signal": [
                        split_index * 10 + label + 0.1,
                        split_index * 10 + label + 0.2,
                    ],
                    "Label": source_labels,
                }
            )
            path = root / relative
            data.to_csv(path, index=False)
            manifest_rows.append(
                {
                    "split": split,
                    "source_partition": (
                        "test" if split == "test" else "train"
                    ),
                    "source_id": f"{split}_source_{label}",
                    "path": relative,
                    "target": label,
                }
            )
    if duplicate_content:
        first = root / "train_0.csv"
        (root / "validation_0.csv").write_bytes(first.read_bytes())
    manifest = root / "manifest.csv"
    pd.DataFrame(manifest_rows).to_csv(manifest, index=False)
    features = root / "features.json"
    features.write_text('["signal"]', encoding="utf-8")
    return manifest, features


def prepare_scania_fixture(root, *, fractional_label=None):
    root = Path(root)
    training_ids = [f"train_vehicle_{i}" for i in range(5)]
    boundaries = [60, 48, 24, 12, 6]
    pd.DataFrame(
        {
            "vehicle_id": training_ids,
            "time_step": [0] * 5,
            "sensor": np.arange(5, dtype=float),
        }
    ).to_csv(root / "train_operational_readouts.csv", index=False)
    pd.DataFrame(
        {
            "vehicle_id": list(reversed(training_ids)),
            "length_of_study_time_step": list(reversed(boundaries)),
            "in_study_repair": [1] * 5,
        }
    ).to_csv(root / "train_tte.csv", index=False)

    for split in ("validation", "test"):
        vehicle_ids = [f"{split}_vehicle_{i}" for i in range(5)]
        pd.DataFrame(
            {
                "vehicle_id": vehicle_ids,
                "time_step": [1] * 5,
                "sensor": np.arange(5, dtype=float),
            }
        ).to_csv(
            root / f"{split}_operational_readouts.csv",
            index=False,
        )
        labels = list(range(5))
        if fractional_label is not None and split == "validation":
            labels[0] = fractional_label
        pd.DataFrame(
            {"vehicle_id": vehicle_ids, "class_label": labels}
        ).to_csv(root / f"{split}_labels.csv", index=False)


class ExperimentValidationTests(unittest.TestCase):
    def test_seed_and_argument_validation(self):
        base = [
            "experiment",
            "--train",
            "train.csv",
            "--validation",
            "validation.csv",
            "--test",
            "test.csv",
            "--features",
            "features.json",
            "--model",
            "logistic",
        ]
        invalid_options = (
            ["--seeds", "7", "7"],
            ["--seeds", "-1"],
            ["--seeds", str(2**32)],
            ["--epochs", "0"],
            ["--batch", "0"],
            ["--window", "1"],
            ["--quantile", "1"],
            ["--jobs", "0"],
            ["--learning-rate", "nan"],
            ["--max-gap", "0", "--model", "lstm_ae", "--group", "g", "--time", "t"],
            ["--target", "Sample_ID"],
        )
        for invalid in invalid_options:
            with self.subTest(options=invalid):
                with tempfile.TemporaryDirectory() as temporary:
                    argv = base + invalid + [
                        "--output",
                        str(Path(temporary) / "new-output"),
                    ]
                    with patch("sys.argv", argv):
                        with self.assertRaises(SystemExit) as caught:
                            EXPERIMENT.parse_arguments()
                    self.assertEqual(caught.exception.code, 2)

    def test_duplicate_ids_and_cross_split_group_overlap(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = fixture_splits(root)
            train = pd.read_csv(args.train)
            train.loc[1, "sample_id"] = train.loc[0, "sample_id"]
            train.to_csv(args.train, index=False)
            with self.assertRaisesRegex(ValueError, "duplicate sample ID"):
                EXPERIMENT.load_data(args)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = fixture_splits(root)
            validation = pd.read_csv(args.validation)
            validation.loc[0, "sample_id"] = "tr0"
            validation.to_csv(args.validation, index=False)
            with self.assertRaisesRegex(ValueError, "sample_id overlaps"):
                EXPERIMENT.load_data(args)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = fixture_splits(root, groups=True)
            validation = pd.read_csv(args.validation)
            validation.loc[0, "sequence_id"] = "train_group_0"
            validation.to_csv(args.validation, index=False)
            with self.assertRaisesRegex(ValueError, "sequence_id overlaps"):
                EXPERIMENT.load_data(args)

    def test_non_integer_and_overflowing_labels_are_rejected_exactly(self):
        for value in ("0.5", "1.0000000000000000001", str(2**63)):
            with self.subTest(value=value), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                args = fixture_splits(root)
                frame = pd.read_csv(args.train, dtype={"target": str})
                frame.loc[0, "target"] = value
                frame.to_csv(args.train, index=False)
                with self.assertRaises(ValueError):
                    EXPERIMENT.load_data(args)

    def test_binary_threshold_requires_both_validation_classes(self):
        with self.assertRaisesRegex(ValueError, "both validation classes"):
            EXPERIMENT.select_threshold(
                np.asarray([0, 0, 0]),
                np.asarray([0.1, 0.2, 0.3]),
            )

    def test_validation_split_must_contain_both_binary_classes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = fixture_splits(root)
            validation = pd.read_csv(args.validation)
            validation["target"] = 0
            validation.to_csv(args.validation, index=False)
            with self.assertRaisesRegex(ValueError, "requires both classes"):
                EXPERIMENT.load_data(args)

    def test_supervised_training_classes_must_be_contiguous(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = fixture_splits(root)
            train = pd.read_csv(args.train)
            train["target"] = [0, 2, 0, 2]
            train.to_csv(args.train, index=False)
            with self.assertRaisesRegex(ValueError, "contiguous"):
                EXPERIMENT.load_data(args)

    def test_threshold_tie_chooses_highest_exact_threshold(self):
        threshold = EXPERIMENT.select_threshold(
            np.asarray([1, 0, 0, 1]),
            np.asarray([0.9, 0.3, 0.2, 0.1]),
        )
        self.assertEqual(threshold, 0.9)

    def test_balanced_accuracy_threshold_is_prevalence_invariant(self):
        target = np.asarray([0, 0, 1, 1])
        score = np.asarray([0.1, 0.4, 0.6, 0.9])
        threshold = EXPERIMENT.select_balanced_accuracy_threshold(
            target,
            score,
        )
        repeated_threshold = (
            EXPERIMENT.select_balanced_accuracy_threshold(
                np.tile(target, 3),
                np.tile(score, 3),
            )
        )
        self.assertEqual(threshold, repeated_threshold)
        self.assertEqual(threshold, 0.6)

    def test_scania_cost_decision_minimizes_expected_cost(self):
        scores = np.asarray([[0.8, 0.2, 0.0, 0.0, 0.0]])
        self.assertEqual(int(scores.argmax(axis=1)[0]), 0)
        self.assertEqual(
            int(EXPERIMENT.scania_cost_decision(scores)[0]),
            1,
        )

    def test_single_class_ranking_metrics_are_null_with_reasons(self):
        result = EXPERIMENT.metrics(
            target=[0, 0, 0],
            prediction=[0, 1, 0],
            score=[0.1, 0.8, 0.2],
            n_classes=2,
        )
        self.assertIsNone(result["roc_auc"])
        self.assertIsNone(result["average_precision"])
        self.assertIn("only class 0", result["undefined_metric_reasons"]["roc_auc"])

    def test_metric_values_and_confusion_orientation(self):
        result = EXPERIMENT.metrics(
            target=[0, 0, 1, 1, 0, 1],
            prediction=[0, 1, 1, 1, 0, 0],
            score=[0.1, 0.4, 0.8, 0.7, 0.9, 0.3],
            n_classes=2,
        )
        self.assertEqual(result["confusion_matrix"], [[2, 1], [1, 2]])
        self.assertAlmostEqual(result["accuracy"], 2 / 3)
        self.assertAlmostEqual(result["balanced_accuracy"], 2 / 3)
        self.assertAlmostEqual(result["precision"], 2 / 3)
        self.assertAlmostEqual(result["recall"], 2 / 3)
        self.assertAlmostEqual(result["f1"], 2 / 3)
        self.assertAlmostEqual(result["macro_f1"], 2 / 3)
        self.assertAlmostEqual(result["fpr"], 1 / 3)
        self.assertAlmostEqual(result["alarm_rate"], 0.5)
        self.assertAlmostEqual(result["prevalence"], 0.5)
        self.assertAlmostEqual(result["roc_auc"], 5 / 9)
        self.assertAlmostEqual(
            result["average_precision"],
            (0.5 + 2 / 3 + 3 / 5) / 3,
        )
        self.assertEqual(result["per_class"]["0"]["support"], 3)

    def test_scania_cost_matrix_and_metrics(self):
        result = EXPERIMENT.metrics(
            target=[0, 1, 2, 3, 4],
            prediction=[0, 0, 1, 2, 3],
            score=None,
            n_classes=5,
            scania=True,
        )
        self.assertEqual(result["total_cost"], 800)
        self.assertEqual(result["mean_cost"], 160.0)

        with self.assertRaisesRegex(ValueError, "exactly five classes"):
            EXPERIMENT.metrics(
                target=[0, 1],
                prediction=[0, 1],
                score=[0.1, 0.9],
                n_classes=2,
                scania=True,
            )

    def test_scania_cost_option_is_accepted_by_cli(self):
        with tempfile.TemporaryDirectory() as temporary:
            argv = [
                "experiment",
                "--train", "train.csv",
                "--validation", "validation.csv",
                "--test", "test.csv",
                "--features", "features.json",
                "--model", "random_forest",
                "--scania-cost",
                "--output", str(Path(temporary) / "output"),
            ]
            with patch("sys.argv", argv):
                args = EXPERIMENT.parse_arguments()
            self.assertTrue(args.scania_cost)

    def test_scania_cost_flag_writes_minimum_expected_cost_predictions(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = fixture_splits(root)
            for split in ("train", "validation", "test"):
                targets = np.tile(np.arange(5), 2)
                pd.DataFrame(
                    {
                        "sample_id": [
                            f"{split}_{index}" for index in range(10)
                        ],
                        "signal": targets.astype(float)
                        + np.tile([0.0, 0.05], 5),
                        "target": targets,
                    }
                ).to_csv(root / f"{split}.csv", index=False)
            output = root / "cost-run"
            run_experiment_cli(
                [
                    sys.executable,
                    str(EXPERIMENT_PATH),
                    "--train",
                    args.train,
                    "--validation",
                    args.validation,
                    "--test",
                    args.test,
                    "--features",
                    args.features,
                    "--model",
                    "logistic",
                    "--scania-cost",
                    "--seeds",
                    "11",
                    "--output",
                    str(output),
                ]
            )
            predictions = pd.read_csv(
                output / "seed_11" / "test_predictions.csv"
            )
            scores = predictions[
                [f"score_{index}" for index in range(5)]
            ].to_numpy()
            expected = EXPERIMENT.scania_cost_decision(scores)
            self.assertEqual(
                predictions["y_pred"].to_list(),
                expected.tolist(),
            )
            decision = json.loads(
                (output / "seed_11" / "decision.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                decision["prediction_rule"],
                "minimum expected cost from class scores",
            )

    def test_endpoint_only_window_index_selects_last_window_per_group(self):
        frame = pd.DataFrame(
            {
                "group": ["a"] * 4 + ["b"] * 4,
                "time": [0, 1, 2, 3] * 2,
            }
        )
        args = Namespace(
            group="group",
            time="time",
            window=2,
            max_gap=None,
            endpoint_only_evaluation=True,
            max_train_windows_per_group=None,
        )
        specification = EXPERIMENT.build_window_index(
            frame,
            np.zeros(len(frame), dtype=int),
            args,
        )
        self.assertEqual(specification["endpoints"].tolist(), [3, 7])
        self.assertEqual(specification["starts"].tolist(), [2, 6])

    def test_lstm_training_window_limit_applies_per_group_after_normal_filter(self):
        frame = pd.DataFrame(
            {
                "group": ["a"] * 6 + ["b"] * 6,
                "time": [0, 1, 2, 3, 4, 5] * 2,
            }
        )
        args = Namespace(
            group="group",
            time="time",
            window=2,
            max_gap=None,
            endpoint_only_evaluation=False,
            max_train_windows_per_group=2,
        )
        specification = EXPERIMENT.build_window_index(
            frame,
            np.zeros(len(frame), dtype=int),
            args,
        )
        self.assertEqual(
            specification["fully_normal_windows_before_limit"],
            10,
        )
        self.assertEqual(int(specification["clean"].sum()), 4)

    def test_metric_labels_reject_fractional_strings(self):
        with self.assertRaisesRegex(ValueError, "integer labels"):
            EXPERIMENT.metrics(
                target=["0", "1.000000000000000001"],
                prediction=[0, 1],
                score=[0.1, 0.9],
                n_classes=2,
            )

    def test_all_missing_eligible_training_feature_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = fixture_splits(root, all_missing=True)
            frames, features, n_classes, _ = EXPERIMENT.load_data(args)
            with self.assertRaisesRegex(ValueError, "completely missing"):
                EXPERIMENT.run_one(
                    args,
                    frames,
                    features,
                    n_classes,
                    11,
                    root / "run",
                )

    def test_autoencoder_preprocessing_uses_normal_training_rows_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = fixture_splits(root)
            for split, labels in (
                ("train", [0, 1, 0, 1]),
                ("validation", [0, 1, 0, 1]),
                ("test", [0, 1, 0, 1]),
            ):
                values = [np.nan, 1.0, np.nan, 2.0] if split == "train" else [
                    0.0,
                    1.0,
                    0.1,
                    0.9,
                ]
                pd.DataFrame(
                    {
                        "sample_id": [f"{split}_{i}" for i in range(4)],
                        "sequence_id": [f"{split}_{i}" for i in range(4)],
                        "timestamp": [0, 1, 0, 1],
                        "signal": values,
                        "target": labels,
                    }
                ).to_csv(root / f"{split}.csv", index=False)
            (root / "features.json").write_text(
                '["signal"]',
                encoding="utf-8",
            )
            args.group = "sequence_id"
            args.time = "timestamp"
            args.model = "lstm_ae"
            frames, features, n_classes, _ = EXPERIMENT.load_data(args)
            with self.assertRaisesRegex(ValueError, "eligible training data"):
                EXPERIMENT.run_one(
                    args,
                    frames,
                    features,
                    n_classes,
                    11,
                    root / "run",
                )

    def test_float32_overflow_fails_after_preprocessing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = fixture_splits(root)
            test = pd.read_csv(args.test)
            test.loc[0, "signal"] = 1e100
            test.to_csv(args.test, index=False)
            frames, features, n_classes, _ = EXPERIMENT.load_data(args)
            with self.assertRaisesRegex(
                ValueError,
                "after float32 conversion",
            ):
                EXPERIMENT.run_one(
                    args,
                    frames,
                    features,
                    n_classes,
                    11,
                    root / "run",
                )

    def test_json_is_strict_and_keeps_numeric_types(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "values.json"
            EXPERIMENT.save_json(path, {"count": np.int64(2), "value": 1.5})
            saved = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(saved, {"count": 2, "value": 1.5})
            with self.assertRaises(ValueError):
                EXPERIMENT.save_json(path, {"bad": float("nan")})

    def test_preparation_metadata_and_feature_hash_are_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = fixture_splits(root)
            (root / "preparation.json").write_text(
                '{"dataset":"fixture","seed":3}',
                encoding="utf-8",
            )
            _, _, _, audit = EXPERIMENT.load_data(args)
            self.assertEqual(
                audit["preparation_metadata"]["metadata"]["dataset"],
                "fixture",
            )
            self.assertEqual(len(audit["features_file"]["sha256"]), 64)

            (root / "preparation.json").write_text(
                '{"dataset":NaN}',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "non-finite value"):
                EXPERIMENT.load_data(args)

    def test_minimal_cpu_classification_run_and_overwrite_protection(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = fixture_splits(root)
            command = [
                sys.executable,
                str(EXPERIMENT_PATH),
                "--train",
                args.train,
                "--validation",
                args.validation,
                "--test",
                args.test,
                "--features",
                args.features,
                "--model",
                "logistic",
                "--device",
                "cpu",
                "--seeds",
                "11",
                "--output",
                args.output,
            ]
            subprocess.run(command, check=True, capture_output=True, text=True)
            output = Path(args.output)
            metrics = json.loads(
                (output / "seed_11" / "metrics.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(metrics["n_test"], 4)
            predictions = pd.read_csv(
                output / "seed_11" / "test_predictions.csv"
            )
            self.assertIn("score_0", predictions.columns)
            self.assertIn("score_1", predictions.columns)
            self.assertEqual(
                predictions["sample_id"].tolist(),
                ["te0", "te1", "te2", "te3"],
            )
            baselines = json.loads(
                (output / "seed_11" / "baselines.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                set(baselines),
                {"train_majority", "always_normal", "always_positive"},
            )
            self.assertTrue((output / "summary.json").is_file())
            with self.assertRaises(subprocess.CalledProcessError) as caught:
                subprocess.run(
                    command,
                    check=True,
                    capture_output=True,
                    text=True,
                )
            self.assertIn("already exists", caught.exception.stderr)
            self.assertTrue((output / "seed_11" / "metrics.json").is_file())

    def test_validate_only_checks_data_without_creating_results(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = fixture_splits(root)
            result = subprocess.run(
                [
                    sys.executable,
                    str(EXPERIMENT_PATH),
                    "--train",
                    args.train,
                    "--validation",
                    args.validation,
                    "--test",
                    args.test,
                    "--features",
                    args.features,
                    "--model",
                    "logistic",
                    "--validate-only",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            report = json.loads(result.stdout)
            self.assertEqual(report["status"], "validated")
            self.assertEqual(report["rows"], {
                "train": 4,
                "validation": 4,
                "test": 4,
            })

    def test_completed_seed_is_preserved_when_a_later_seed_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "results"
            args = Namespace(
                validate_only=False,
                scania_cost=False,
                output=str(output),
                seeds=[11, 22],
            )
            with (
                patch.object(EXPERIMENT, "parse_arguments", return_value=args),
                patch.object(
                    EXPERIMENT,
                    "load_data",
                    return_value=(
                        {"train": pd.DataFrame(), "validation": pd.DataFrame(), "test": pd.DataFrame()},
                        ["signal"],
                        2,
                        {},
                    ),
                ),
                patch.object(EXPERIMENT, "write_environment_files"),
                patch.object(
                    EXPERIMENT,
                    "run_one",
                    side_effect=[
                        {"seed": 11, "f1": 0.5},
                        ValueError("fixture failure"),
                    ],
                ),
            ):
                with self.assertRaisesRegex(ValueError, "fixture failure"):
                    EXPERIMENT.main()

            status = json.loads(
                (output / "status.json").read_text(encoding="utf-8")
            )
            self.assertEqual(status["status"], "failed")
            self.assertEqual(status["completed_seeds"], [11])
            self.assertTrue((output / "runs.csv").is_file())

    def test_keyboard_interrupt_is_recorded_and_reraised(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "results"
            args = Namespace(
                validate_only=False,
                scania_cost=False,
                output=str(output),
                seeds=[11],
            )
            with (
                patch.object(EXPERIMENT, "parse_arguments", return_value=args),
                patch.object(
                    EXPERIMENT,
                    "load_data",
                    return_value=(
                        {"train": pd.DataFrame(), "validation": pd.DataFrame(), "test": pd.DataFrame()},
                        ["signal"],
                        2,
                        {},
                    ),
                ),
                patch.object(EXPERIMENT, "write_environment_files"),
                patch.object(
                    EXPERIMENT,
                    "run_one",
                    side_effect=KeyboardInterrupt(),
                ),
            ):
                with self.assertRaises(KeyboardInterrupt):
                    EXPERIMENT.main()

            status = json.loads(
                (output / "status.json").read_text(encoding="utf-8")
            )
            self.assertEqual(status["status"], "interrupted")
            self.assertEqual(status["completed_seeds"], [])


class WindowTests(unittest.TestCase):
    def make_args(self, window=2, max_gap=None):
        return Namespace(group="group", time="time", window=window, max_gap=max_gap)

    def test_lazy_contents_normal_filter_and_endpoint_alignment(self):
        frame = pd.DataFrame(
            {
                "sample_id": ["g0", "g1", "g2", "g3"],
                "group": ["g", "g", "g", "g"],
                "time": [0, 1, 2, 3],
            }
        )
        values = np.arange(8, dtype=np.float32).reshape(4, 2)
        targets = np.asarray([0, 0, 1, 0])
        specification = EXPERIMENT.build_window_index(
            frame,
            targets,
            self.make_args(),
        )
        windows = EXPERIMENT.LazyWindowView(
            values,
            specification,
            2,
            mask=specification["clean"],
        )
        self.assertEqual(len(windows), 1)
        np.testing.assert_array_equal(windows[0], values[:2])
        np.testing.assert_array_equal(specification["endpoints"], [1, 2, 3])
        self.assertEqual(
            frame.iloc[specification["endpoints"]]["sample_id"].tolist(),
            ["g1", "g2", "g3"],
        )
        np.testing.assert_array_equal(targets[specification["endpoints"]], [0, 1, 0])
        self.assertIs(windows.values, values)

    def test_windows_do_not_cross_groups_and_sort_by_time(self):
        frame = pd.DataFrame(
            {
                "group": ["a", "a", "b", "b"],
                "time": [1, 0, 0, 1],
            }
        )
        specification = EXPERIMENT.build_window_index(
            frame,
            np.zeros(4, dtype=int),
            self.make_args(),
        )
        np.testing.assert_array_equal(specification["order"], [1, 0, 2, 3])
        np.testing.assert_array_equal(specification["endpoints"], [0, 3])
        self.assertEqual(len(specification["starts"]), 2)

    def test_duplicate_timestamps_fail(self):
        frame = pd.DataFrame({"group": ["g", "g"], "time": [1, 1]})
        with self.assertRaisesRegex(ValueError, "strictly increase"):
            EXPERIMENT.build_window_index(
                frame,
                np.zeros(2, dtype=int),
                self.make_args(),
            )

    def test_max_gap_splits_sequences_before_windowing(self):
        frame = pd.DataFrame(
            {"group": ["g"] * 5, "time": [0, 1, 2, 10, 11]}
        )
        specification = EXPERIMENT.build_window_index(
            frame,
            np.zeros(5, dtype=int),
            self.make_args(max_gap=2),
        )
        np.testing.assert_array_equal(specification["endpoints"], [1, 2, 4])
        self.assertEqual(len(specification["starts"]), 3)

    def test_short_sequence_has_helpful_no_window_error(self):
        frame = pd.DataFrame({"group": ["g"], "time": [0]})
        with self.assertRaisesRegex(ValueError, "No complete LSTM windows"):
            EXPERIMENT.build_window_index(
                frame,
                np.zeros(1, dtype=int),
                self.make_args(),
            )

    def test_lazy_view_rejects_misaligned_normal_mask(self):
        with self.assertRaisesRegex(ValueError, "mask does not match"):
            EXPERIMENT.LazyWindowView(
                np.zeros((2, 1), dtype=np.float32),
                {"order": np.arange(2), "starts": np.asarray([0])},
                2,
                mask=np.asarray([True, False]),
            )


class PreparationTests(unittest.TestCase):
    def test_ciciomt_preparer_writes_fixture_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, features = write_ciciomt_fixture(root)
            output = root / "prepared"
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "prepare_ciciomt.py"),
                    "--root",
                    str(root),
                    "--manifest",
                    str(manifest),
                    "--features",
                    str(features),
                    "--output",
                    str(output),
                    "--chunksize",
                    "1",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            prepared = pd.read_csv(output / "train.csv")
            self.assertEqual(prepared["target"].value_counts().to_dict(), {0: 2, 1: 2})
            metadata = json.loads(
                (output / "preparation.json").read_text(encoding="utf-8")
            )
            self.assertEqual(len(metadata["files"]), 6)

    def test_ciciomt_preparer_rejects_mixed_label_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, features = write_ciciomt_fixture(root, mixed=True)
            command = [
                sys.executable,
                str(ROOT / "scripts" / "prepare_ciciomt.py"),
                "--root",
                str(root),
                "--manifest",
                str(manifest),
                "--features",
                str(features),
                "--output",
                str(root / "prepared"),
            ]
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("mixed labels", result.stderr)

    def test_ciciomt_preparer_rejects_duplicate_file_content(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, features = write_ciciomt_fixture(
                root,
                duplicate_content=True,
            )
            command = [
                sys.executable,
                str(ROOT / "scripts" / "prepare_ciciomt.py"),
                "--root",
                str(root),
                "--manifest",
                str(manifest),
                "--features",
                str(features),
                "--output",
                str(root / "prepared"),
            ]
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Duplicate source-file content", result.stderr)

    def test_ciciomt_preparer_rejects_nonpositive_chunksize(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, features = write_ciciomt_fixture(root)
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "prepare_ciciomt.py"),
                    "--root",
                    str(root),
                    "--manifest",
                    str(manifest),
                    "--features",
                    str(features),
                    "--output",
                    str(root / "prepared"),
                    "--chunksize",
                    "0",
                ],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("positive integer", result.stderr)

    def test_ciciomt_validation_must_use_training_partition(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, features = write_ciciomt_fixture(root)
            table = pd.read_csv(manifest)
            table.loc[table["split"] == "validation", "source_partition"] = "test"
            table.to_csv(manifest, index=False)
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "prepare_ciciomt.py"),
                    "--root",
                    str(root),
                    "--manifest",
                    str(manifest),
                    "--features",
                    str(features),
                    "--output",
                    str(root / "prepared"),
                ],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("official training partition", result.stderr)

    def test_ciciomt_validation_accepts_profiling_partition(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, features = write_ciciomt_fixture(root)
            table = pd.read_csv(manifest)
            table.loc[
                table["split"] == "validation",
                "source_partition",
            ] = "profiling"
            table.to_csv(manifest, index=False)
            output = root / "prepared"
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "prepare_ciciomt.py"),
                    "--root",
                    str(root),
                    "--manifest",
                    str(manifest),
                    "--features",
                    str(features),
                    "--output",
                    str(output),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            metadata = json.loads(
                (output / "preparation.json").read_text(encoding="utf-8")
            )
            self.assertTrue(
                all(
                    item["source_partition"] == "profiling"
                    for item in metadata["files"]
                    if item["split"] == "validation"
                )
            )

    def test_ciciomt_can_split_profiling_captures_between_train_and_validation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, features = write_ciciomt_fixture(root)
            table = pd.read_csv(manifest)
            for index in range(2):
                relative = f"profile_{index}.csv"
                pd.DataFrame(
                    {"signal": [index + 0.25], "Label": [0]}
                ).to_csv(root / relative, index=False)
                table.loc[len(table)] = {
                    "split": "validation",
                    "source_partition": "profiling",
                    "source_id": f"profile_capture_{index}",
                    "path": relative,
                    "target": 0,
                }
            table.to_csv(manifest, index=False)
            output = root / "prepared"
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "prepare_ciciomt.py"),
                    "--root",
                    str(root),
                    "--manifest",
                    str(manifest),
                    "--features",
                    str(features),
                    "--output",
                    str(output),
                    "--profiling-train-fraction",
                    "0.5",
                    "--seed",
                    "7",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            metadata = json.loads(
                (output / "preparation.json").read_text(encoding="utf-8")
            )
            profiling = [
                (item["source_id"], item["split"])
                for item in metadata["files"]
                if item["source_partition"] == "profiling"
            ]
            self.assertCountEqual(
                [split for _, split in profiling],
                ["train", "validation"],
            )
            self.assertEqual(
                metadata["profiling_split"]["captures_moved_to_train"],
                1,
            )
            self.assertEqual(
                metadata["profiling_split"][
                    "captures_held_out_for_validation"
                ],
                1,
            )

    def test_scania_preparer_joins_event_data_by_vehicle_id(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prepare_scania_fixture(root)
            for split in ("validation", "test"):
                path = root / f"{split}_operational_readouts.csv"
                readouts = pd.read_csv(path)
                earlier = readouts.copy()
                earlier["time_step"] = 0
                earlier["sensor"] += 0.5
                pd.concat([earlier, readouts], ignore_index=True).to_csv(
                    path,
                    index=False,
                )
            output = root / "processed"
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "prepare_scania.py"),
                    "--root",
                    str(root),
                    "--output",
                    str(output),
                    "--seed",
                    "3",
                    "--write-anomaly-sequences",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            training = pd.read_csv(output / "train.csv")
            targets = training.set_index("vehicle_id")["target"]
            expected = {
                "train_vehicle_0": 0,
                "train_vehicle_1": 1,
                "train_vehicle_2": 2,
                "train_vehicle_3": 3,
                "train_vehicle_4": 4,
            }
            self.assertEqual(targets.to_dict(), expected)
            metadata = json.loads(
                (output / "preparation.json").read_text(encoding="utf-8")
            )
            self.assertEqual(metadata["class_boundaries"]["4"], "remaining <= 6")
            validation_sequences = pd.read_csv(
                output / "validation_anomaly_sequences.csv"
            )
            self.assertEqual(len(validation_sequences), 10)
            self.assertEqual(
                validation_sequences.groupby("vehicle_id").size().unique().tolist(),
                [2],
            )
            self.assertEqual(
                validation_sequences["anomaly_target"].value_counts().to_dict(),
                {1: 8, 0: 2},
            )
            self.assertEqual(
                len(pd.read_csv(output / "validation.csv")),
                5,
            )

    def test_scania_preparer_can_keep_all_eligible_training_readouts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prepare_scania_fixture(root)
            readouts_path = root / "train_operational_readouts.csv"
            readouts = pd.read_csv(readouts_path)
            later_readouts = readouts.copy()
            later_readouts["time_step"] = 1
            pd.concat(
                [readouts, later_readouts],
                ignore_index=True,
            ).to_csv(readouts_path, index=False)
            output = root / "processed"
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "prepare_scania.py"),
                    "--root",
                    str(root),
                    "--output",
                    str(output),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            training = pd.read_csv(output / "train.csv")
            self.assertEqual(len(training), 10)
            self.assertEqual(
                training["target"].value_counts().sort_index().to_dict(),
                {0: 2, 1: 2, 2: 2, 3: 2, 4: 2},
            )
            self.assertEqual(training["sample_id"].nunique(), 10)
            metadata = json.loads(
                (output / "preparation.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                metadata["training_sampling"],
                "all_eligible_readouts",
            )

    def test_scania_preparer_rejects_out_of_range_seed_and_fractional_label(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prepare_scania_fixture(root)
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "prepare_scania.py"),
                    "--root",
                    str(root),
                    "--output",
                    str(root / "bad-seed"),
                    "--seed",
                    str(2**32),
                ],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("0 <= seed < 2**32", result.stderr)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prepare_scania_fixture(root, fractional_label="1.000000000000000001")
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "prepare_scania.py"),
                    "--root",
                    str(root),
                    "--output",
                    str(root / "fractional-label"),
                ],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("invalid class labels", result.stderr)


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
CUDA_AVAILABLE = False
if TORCH_AVAILABLE:
    import torch

    CUDA_AVAILABLE = torch.cuda.is_available()


class OptionalTorchSmokeTests(unittest.TestCase):
    @unittest.skipUnless(
        not TORCH_AVAILABLE,
        "PyTorch is installed; CUDA availability is tested by the CUDA smoke test",
    )
    def test_cuda_request_without_torch_fails_clearly(self):
        args = Namespace(device="cuda", model="mlp")
        with self.assertRaisesRegex(RuntimeError, "PyTorch is not installed"):
            EXPERIMENT.neural_model(
                args,
                {},
                {},
                2,
                {},
                11,
                Path("unused"),
                [],
            )

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed")
    def test_mlp_cpu_smoke(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = fixture_splits(root)
            args.model = "mlp"
            args.device = "cpu"
            args.output = str(root / "mlp-output")
            run_experiment_cli(
                [
                    sys.executable,
                    str(EXPERIMENT_PATH),
                    "--train",
                    args.train,
                    "--validation",
                    args.validation,
                    "--test",
                    args.test,
                    "--features",
                    args.features,
                    "--model",
                    "mlp",
                    "--device",
                    "cpu",
                    "--epochs",
                    "1",
                    "--batch",
                    "2",
                    "--hidden",
                    "4",
                    "--seeds",
                    "11",
                    "--output",
                    args.output,
                ]
            )

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed")
    def test_lstm_autoencoder_cpu_smoke(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            targets_and_groups = {
                "train": ([0, 0, 0, 0, 0, 0], ["tr_a"] * 3 + ["tr_b"] * 3),
                "validation": (
                    [0, 0, 0, 0, 1, 0],
                    ["va_a"] * 3 + ["va_b"] * 3,
                ),
                "test": ([0, 0, 1, 0, 1, 1], ["te_a"] * 3 + ["te_b"] * 3),
            }
            for split, (targets, groups) in targets_and_groups.items():
                pd.DataFrame(
                    {
                        "sample_id": [
                            f"{split}_{index}" for index in range(6)
                        ],
                        "sequence_id": groups,
                        "time_step": [0, 1, 2, 0, 1, 2],
                        "signal": [0.1, 0.2, 0.3, 1.0, 1.1, 1.2],
                        "target": targets,
                    }
                ).to_csv(root / f"{split}.csv", index=False)
            features = root / "features.json"
            features.write_text('["signal"]', encoding="utf-8")
            output = root / "lstm-output"

            run_experiment_cli(
                [
                    sys.executable,
                    str(EXPERIMENT_PATH),
                    "--train",
                    str(root / "train.csv"),
                    "--validation",
                    str(root / "validation.csv"),
                    "--test",
                    str(root / "test.csv"),
                    "--features",
                    str(features),
                    "--model",
                    "lstm_ae",
                    "--device",
                    "cpu",
                    "--group",
                    "sequence_id",
                    "--time",
                    "time_step",
                    "--window",
                    "2",
                    "--epochs",
                    "1",
                    "--batch",
                    "2",
                    "--hidden",
                    "4",
                    "--latent",
                    "2",
                    "--seeds",
                    "11",
                    "--output",
                    str(output),
                ]
            )

            metrics = json.loads(
                (output / "seed_11" / "metrics.json").read_text(
                    encoding="utf-8"
                )
            )
            predictions = pd.read_csv(
                output / "seed_11" / "test_predictions.csv"
            )
            status = json.loads(
                (output / "status.json").read_text(encoding="utf-8")
            )
            self.assertEqual(metrics["device"], "cpu")
            self.assertIsNotNone(metrics["best_epoch"])
            self.assertEqual(metrics["n_test"], 4)
            self.assertEqual(
                predictions["sample_id"].tolist(),
                ["test_1", "test_2", "test_4", "test_5"],
            )
            self.assertEqual(predictions["y_true"].tolist(), [0, 1, 1, 1])
            self.assertEqual(status["status"], "completed")

    @unittest.skipUnless(
        TORCH_AVAILABLE and not CUDA_AVAILABLE,
        "Requires an installed CPU-only PyTorch build",
    )
    def test_cuda_request_with_cpu_only_torch_fails_clearly(self):
        args = Namespace(device="cuda", model="mlp")
        with self.assertRaisesRegex(
            RuntimeError,
            "CUDA requested but unavailable",
        ):
            EXPERIMENT.neural_model(
                args,
                {},
                {},
                2,
                {},
                11,
                Path("unused"),
                [],
            )

    @unittest.skipUnless(
        CUDA_AVAILABLE,
        "CUDA test skipped: PyTorch CUDA is unavailable",
    )
    def test_mlp_cuda_smoke(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = fixture_splits(root)
            args.output = str(root / "cuda-output")
            run_experiment_cli(
                [
                    sys.executable,
                    str(EXPERIMENT_PATH),
                    "--train",
                    args.train,
                    "--validation",
                    args.validation,
                    "--test",
                    args.test,
                    "--features",
                    args.features,
                    "--model",
                    "mlp",
                    "--device",
                    "cuda",
                    "--epochs",
                    "1",
                    "--batch",
                    "2",
                    "--hidden",
                    "4",
                    "--seeds",
                    "11",
                    "--output",
                    args.output,
                ]
            )

            output = Path(args.output)
            seed_output = output / "seed_11"
            metrics = json.loads(
                (seed_output / "metrics.json").read_text(
                    encoding="utf-8"
                )
            )
            predictions = pd.read_csv(
                seed_output / "test_predictions.csv"
            )
            status = json.loads(
                (output / "status.json").read_text(
                    encoding="utf-8"
                )
            )

            self.assertEqual(metrics["device"], "cuda")
            self.assertEqual(
                metrics["gpu"],
                torch.cuda.get_device_name(0),
            )
            self.assertEqual(
                metrics["cuda_runtime"],
                torch.version.cuda,
            )
            self.assertEqual(metrics["n_test"], len(predictions))
            self.assertTrue(
                np.isfinite(
                    predictions[
                        ["score_0", "score_1"]
                    ].to_numpy()
                ).all()
            )
            self.assertTrue((seed_output / "model.pt").is_file())
            self.assertEqual(status["status"], "completed")
            self.assertEqual(status["completed_seeds"], [11])

    @unittest.skipUnless(
        CUDA_AVAILABLE,
        "CUDA test skipped: PyTorch CUDA is unavailable",
    )
    def test_lstm_autoencoder_cuda_smoke(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            targets_and_groups = {
                "train": ([0, 0, 0, 0, 0, 0], ["tr_a"] * 3 + ["tr_b"] * 3),
                "validation": (
                    [0, 0, 0, 0, 1, 0],
                    ["va_a"] * 3 + ["va_b"] * 3,
                ),
                "test": ([0, 0, 1, 0, 1, 1], ["te_a"] * 3 + ["te_b"] * 3),
            }
            for split, (targets, groups) in targets_and_groups.items():
                pd.DataFrame(
                    {
                        "sample_id": [
                            f"{split}_{index}" for index in range(6)
                        ],
                        "sequence_id": groups,
                        "time_step": [0, 1, 2, 0, 1, 2],
                        "signal": [0.1, 0.2, 0.3, 1.0, 1.1, 1.2],
                        "target": targets,
                    }
                ).to_csv(root / f"{split}.csv", index=False)
            features = root / "features.json"
            features.write_text('["signal"]', encoding="utf-8")
            output = root / "lstm-cuda-output"

            run_experiment_cli(
                [
                    sys.executable,
                    str(EXPERIMENT_PATH),
                    "--train",
                    str(root / "train.csv"),
                    "--validation",
                    str(root / "validation.csv"),
                    "--test",
                    str(root / "test.csv"),
                    "--features",
                    str(features),
                    "--model",
                    "lstm_ae",
                    "--device",
                    "cuda",
                    "--group",
                    "sequence_id",
                    "--time",
                    "time_step",
                    "--window",
                    "2",
                    "--epochs",
                    "1",
                    "--batch",
                    "2",
                    "--hidden",
                    "4",
                    "--latent",
                    "2",
                    "--seeds",
                    "11",
                    "--output",
                    str(output),
                ]
            )

            seed_output = output / "seed_11"
            metrics = json.loads(
                (seed_output / "metrics.json").read_text(
                    encoding="utf-8"
                )
            )
            predictions = pd.read_csv(
                seed_output / "test_predictions.csv"
            )
            status = json.loads(
                (output / "status.json").read_text(
                    encoding="utf-8"
                )
            )

            self.assertEqual(metrics["device"], "cuda")
            self.assertEqual(
                metrics["gpu"],
                torch.cuda.get_device_name(0),
            )
            self.assertEqual(
                metrics["cuda_runtime"],
                torch.version.cuda,
            )
            self.assertEqual(metrics["n_test"], len(predictions))
            self.assertTrue(
                np.isfinite(
                    predictions["score"].to_numpy()
                ).all()
            )
            self.assertTrue((seed_output / "model.pt").is_file())
            self.assertEqual(status["status"], "completed")
            self.assertEqual(status["completed_seeds"], [11])


if __name__ == "__main__":
    unittest.main()
