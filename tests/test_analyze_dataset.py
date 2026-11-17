from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.analyze_dataset import analyze_dataset


class DatasetAnalysisTests(unittest.TestCase):
    def create_dataset(self, root: Path) -> Path:
        dataset = root / "prepared"
        dataset.mkdir()
        (dataset / "features.json").write_text(
            '["sensor_a", "sensor_b"]',
            encoding="utf-8",
        )
        for split, labels in (
            ("train", [0, 1, 0, 1]),
            ("validation", [0, 1]),
            ("test", [1, 0, 1]),
        ):
            pd.DataFrame(
                {
                    "sensor_a": [1.0, 2.0, None, 4.0][:len(labels)],
                    "sensor_b": [4.0, float("inf"), 2.0, 1.0][:len(labels)],
                    "target": labels,
                }
            ).to_csv(dataset / f"{split}.csv", index=False)
        return dataset

    def test_analysis_writes_full_data_counts_sample_stats_and_plots(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = self.create_dataset(root)
            output = root / "analysis"

            report = analyze_dataset(
                dataset,
                output,
                sample_size=2,
                chunksize=2,
            )

            self.assertEqual(report["splits"]["train"]["rows"], 4)
            self.assertEqual(
                report["splits"]["train"]["target_distribution"],
                {"0": 2, "1": 2},
            )
            self.assertEqual(
                report["splits"]["train"]["target_distribution_percent"],
                {"0": 50.0, "1": 50.0},
            )
            self.assertEqual(
                report["splits"]["train"]["missing_feature_values"],
                1,
            )
            self.assertEqual(
                report["splits"]["train"]["infinite_feature_values"],
                1,
            )
            self.assertEqual(
                report["splits"]["train"]["sampled_rows"],
                2,
            )
            self.assertEqual(len(report["plots"]), 4)
            for name in report["plots"]:
                self.assertGreater((output / name).stat().st_size, 0)
            saved = json.loads(
                (output / "analysis.json").read_text(encoding="utf-8")
            )
            self.assertEqual(saved["features"], ["sensor_a", "sensor_b"])

    def test_analysis_refuses_existing_output_and_invalid_sample_size(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = self.create_dataset(root)
            output = root / "analysis"
            output.mkdir()

            with self.assertRaisesRegex(FileExistsError, "already exists"):
                analyze_dataset(dataset, output)
            with self.assertRaisesRegex(ValueError, "must be positive"):
                analyze_dataset(dataset, root / "new-analysis", sample_size=0)


if __name__ == "__main__":
    unittest.main()
