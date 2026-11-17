from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from scripts.run_gui import (
    PreparedDataset,
    RunOptions,
    build_ciciomt_preparation_command,
    build_analysis_command,
    build_analysis_output_path,
    build_experiment_command,
    build_scania_preparation_command,
    build_run_output_path,
    discover_prepared_datasets,
    inspect_prepared_dataset,
    resolve_device,
)


class LauncherDatasetTests(unittest.TestCase):
    def create_dataset(self, root: Path) -> Path:
        dataset = root / "scania" / "prepared_v1"
        dataset.mkdir(parents=True)
        (dataset / "features.json").write_text(
            '["sensor_a", "sensor_b"]',
            encoding="utf-8",
        )
        (dataset / "preparation.json").write_text(
            json.dumps(
                {
                    "dataset": "SCANIA Component X",
                    "task": "five-class classification",
                }
            ),
            encoding="utf-8",
        )
        header = "sample_id,vehicle_id,time_step,sensor_a,sensor_b,target\n"
        for split in ("train", "validation", "test"):
            (dataset / f"{split}.csv").write_text(
                header,
                encoding="utf-8",
            )
        return dataset

    def test_dataset_inspection_reads_real_provenance_and_headers(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = self.create_dataset(Path(temporary))
            dataset = inspect_prepared_dataset(path)

            self.assertEqual(dataset.title, "SCANIA Component X")
            self.assertEqual(dataset.task, "five-class classification")
            self.assertEqual(dataset.features, ("sensor_a", "sensor_b"))
            self.assertIn("vehicle_id", dataset.headers["train"])

    def test_discovery_only_returns_complete_prepared_datasets(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            expected = self.create_dataset(root)
            incomplete = root / "partial"
            incomplete.mkdir()
            (incomplete / "features.json").write_text(
                '["sensor"]',
                encoding="utf-8",
            )

            self.assertEqual(
                discover_prepared_datasets(root),
                [expected.resolve()],
            )

    def test_invalid_feature_manifest_fails_explicitly(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = self.create_dataset(Path(temporary))
            (path / "features.json").write_text(
                '["sensor", "Sensor"]',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "уникальных"):
                inspect_prepared_dataset(path)

    def test_scania_preparation_command_uses_project_preparer_arguments(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            command = build_scania_preparation_command(
                "python",
                root / "prepare_scania.py",
                root / "raw",
                root / "prepared",
                42,
            )

            self.assertEqual(
                command,
                [
                    "python",
                    str(root / "prepare_scania.py"),
                    "--root",
                    str(root / "raw"),
                    "--output",
                    str(root / "prepared"),
                    "--seed",
                    "42",
                ],
            )
            with self.assertRaisesRegex(ValueError, "seed"):
                build_scania_preparation_command(
                    "python",
                    root / "prepare_scania.py",
                    root / "raw",
                    root / "prepared",
                    2**32,
                )

    def test_ciciomt_preparation_requires_user_supplied_config(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "csv"
            source.mkdir()
            manifest = root / "manifest.csv"
            features = root / "features.json"
            manifest.write_text("split,path,target\n", encoding="utf-8")
            features.write_text('["signal"]', encoding="utf-8")
            command = build_ciciomt_preparation_command(
                "python",
                root / "prepare_ciciomt.py",
                source,
                manifest,
                features,
                root / "prepared",
            )

            self.assertEqual(
                command,
                [
                    "python",
                    str(root / "prepare_ciciomt.py"),
                    "--root",
                    str(source),
                    "--manifest",
                    str(manifest),
                    "--features",
                    str(features),
                    "--output",
                    str(root / "prepared"),
                ],
            )
            with self.assertRaisesRegex(ValueError, "Манифест не найден"):
                build_ciciomt_preparation_command(
                    "python",
                    root / "prepare_ciciomt.py",
                    source,
                    root / "missing.csv",
                    features,
                    root / "prepared",
                )

    def test_ciciomt_gui_groups_profiling_captures_when_manifest_has_them(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "csv"
            source.mkdir()
            manifest = root / "manifest.csv"
            manifest.write_text(
                "split,source_partition\nvalidation,profiling\n",
                encoding="utf-8",
            )
            features = root / "features.json"
            features.write_text('["signal"]', encoding="utf-8")

            command = build_ciciomt_preparation_command(
                "python",
                root / "prepare_ciciomt.py",
                source,
                manifest,
                features,
                root / "prepared",
            )

            self.assertEqual(
                command[
                    command.index("--profiling-train-fraction") :
                ],
                [
                    "--profiling-train-fraction",
                    "0.8",
                    "--seed",
                    "42",
                ],
            )


class LauncherCommandTests(unittest.TestCase):
    def create_dataset(self, root: Path) -> PreparedDataset:
        directory = root / "prepared"
        directory.mkdir()
        (directory / "features.json").write_text(
            '["signal"]',
            encoding="utf-8",
        )
        headers = {}
        for split in ("train", "validation", "test"):
            (directory / f"{split}.csv").write_text(
                "sample_id,sequence_id,time_step,signal,target\n",
                encoding="utf-8",
            )
            headers[split] = (
                "sample_id",
                "sequence_id",
                "time_step",
                "signal",
                "target",
            )
        return PreparedDataset(
            directory=directory,
            title="test dataset",
            task="test-only fixture",
            features=("signal",),
            headers=headers,
            preparation={},
        )

    def test_cuda_mlp_command_uses_selected_dataset_and_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = self.create_dataset(root)
            output = root / "user-chosen" / "run-001"
            command = build_experiment_command(
                "python.exe",
                Path("iot_experiment.py.py"),
                dataset,
                RunOptions(
                    model="mlp",
                    device="cuda",
                    output=output,
                    seeds=(11, 22),
                    target="target",
                    sample_id="sample_id",
                    epochs=3,
                    batch=8,
                    hidden=16,
                    learning_rate=0.01,
                ),
            )

            self.assertIn("--device", command)
            self.assertEqual(command[command.index("--device") + 1], "cuda")
            self.assertEqual(
                command[command.index("--output") + 1],
                str(output),
            )
            self.assertEqual(
                command[command.index("--train") + 1],
                str(dataset.directory / "train.csv"),
            )
            self.assertEqual(
                command[command.index("--seeds") + 1 : command.index("--epochs")],
                ["11", "22"],
            )

    def test_analysis_command_uses_selected_dataset_and_target(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = self.create_dataset(root)
            output = root / "analysis"
            command = build_analysis_command(
                "python.exe",
                Path("scripts/analyze_dataset.py"),
                dataset,
                output,
                "target",
            )

            self.assertEqual(command[command.index("--dataset") + 1], str(dataset.directory))
            self.assertEqual(command[command.index("--output") + 1], str(output))
            self.assertEqual(command[command.index("--target") + 1], "target")
            self.assertEqual(
                build_analysis_output_path(
                    dataset,
                    datetime(2026, 11, 2, 0, 0, 0),
                ).name,
                "analysis_20261102_000000",
            )

    def test_validate_only_does_not_set_an_output_folder(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = self.create_dataset(root)
            command = build_experiment_command(
                "python.exe",
                Path("iot_experiment.py.py"),
                dataset,
                RunOptions(
                    model="logistic",
                    device="cpu",
                    output=root / "results",
                    seeds=(7,),
                    target="target",
                    sample_id="sample_id",
                ),
                validate_only=True,
            )

            self.assertIn("--validate-only", command)
            self.assertNotIn("--output", command)

    def test_existing_results_and_gpu_on_cpu_model_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = self.create_dataset(root)
            existing = root / "already-there"
            existing.mkdir()
            options = RunOptions(
                model="logistic",
                device="cpu",
                output=existing,
                seeds=(7,),
                target="target",
                sample_id="sample_id",
            )
            with self.assertRaisesRegex(ValueError, "уже существует"):
                build_experiment_command(
                    "python.exe",
                    Path("iot_experiment.py.py"),
                    dataset,
                    options,
                )

            with self.assertRaisesRegex(ValueError, "только на CPU"):
                build_experiment_command(
                    "python.exe",
                    Path("iot_experiment.py.py"),
                    dataset,
                    RunOptions(
                        model="random_forest",
                        device="cuda",
                        output=root / "new-results",
                        seeds=(7,),
                        target="target",
                        sample_id="sample_id",
                    ),
                )

    def test_lstm_command_requires_group_and_time(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = self.create_dataset(root)
            with self.assertRaisesRegex(ValueError, "group и time"):
                build_experiment_command(
                    "python.exe",
                    Path("iot_experiment.py.py"),
                    dataset,
                    RunOptions(
                        model="lstm_ae",
                        device="cpu",
                        output=root / "new-results",
                        seeds=(7,),
                        target="target",
                        sample_id="sample_id",
                    ),
                )

    def test_device_selection_matches_visible_cpu_and_cuda_choices(self):
        self.assertEqual(resolve_device("mlp", "GPU (CUDA)", True), "cuda")
        self.assertEqual(resolve_device("mlp", "CPU", False), "cpu")
        self.assertEqual(
            resolve_device("logistic", "CPU", False),
            "cpu",
        )
        with self.assertRaisesRegex(ValueError, "CUDA недоступна"):
            resolve_device("mlp", "GPU (CUDA)", False)
        with self.assertRaisesRegex(ValueError, "Неизвестный выбор"):
            resolve_device("mlp", "cuda", True)

    def test_run_name_cannot_escape_selected_output_folder(self):
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary) / "chosen"
            parent.mkdir()
            self.assertEqual(
                build_run_output_path(parent, "new-run"),
                parent / "new-run",
            )
            for name in ("..", ".", "../outside", "subfolder/run"):
                with self.subTest(name=name):
                    with self.assertRaises(ValueError):
                        build_run_output_path(parent, name)


if __name__ == "__main__":
    unittest.main()
