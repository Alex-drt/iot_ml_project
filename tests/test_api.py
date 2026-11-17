from __future__ import annotations

import asyncio
import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    import httpx

    from scripts import api
except ImportError:
    httpx = None
    api = None


@unittest.skipIf(api is None, "Install API dependencies from requirements/api.txt")
class ApiTests(unittest.TestCase):
    async def request(self, method, path, **kwargs):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=api.app),
            base_url="http://test",
        ) as client:
            return await client.request(method, path, **kwargs)

    def test_health_and_openapi_routes(self):
        response = asyncio.run(self.request("GET", "/health"))
        schema_response = asyncio.run(self.request("GET", "/openapi.json"))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})
        schema = schema_response.json()
        self.assertIn("/v1/validate", schema["paths"])
        self.assertIn("/v1/experiments", schema["paths"])
        self.assertIn("/v1/predict", schema["paths"])

    def test_validation_jobs_and_path_restriction(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            processed = root / "datasets" / "processed"
            dataset = processed / "fixture"
            dataset.mkdir(parents=True)
            outside = root / "outside"
            outside.mkdir()
            with (
                patch.object(api, "API_PROCESSED", processed),
                patch.object(api, "_run_command", return_value={"stdout": "validated"}),
            ):
                response = asyncio.run(
                    self.request(
                        "POST",
                        "/v1/validate",
                        json={"dataset": str(dataset)},
                    )
                )
                self.assertEqual(response.status_code, 202)
                job_id = response.json()["job_id"]
                for _ in range(100):
                    job = asyncio.run(
                        self.request("GET", f"/v1/jobs/{job_id}")
                    ).json()
                    if job["status"] in {"completed", "failed"}:
                        break
                    time.sleep(0.01)
                self.assertEqual(job["status"], "completed")
                self.assertEqual(job["result"]["stdout"], "validated")
                rejected = asyncio.run(
                    self.request(
                        "POST",
                        "/v1/validate",
                        json={"dataset": str(outside)},
                    )
                )
                self.assertEqual(rejected.status_code, 400)

    def test_predict_from_saved_logistic_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root / "results" / "fixture-run"
            seed = run / "seed_42"
            seed.mkdir(parents=True)
            training = np.asarray([[0.0], [0.2], [0.8], [1.0]])
            labels = np.asarray([0, 0, 1, 1])
            preprocessing = Pipeline(
                [
                    ("scaler", StandardScaler()),
                ]
            ).fit(training)
            model = LogisticRegression().fit(
                preprocessing.transform(training),
                labels,
            )
            joblib.dump(preprocessing, seed / "preprocessing.joblib")
            joblib.dump(model, seed / "model.joblib")
            (seed / "features.json").write_text(
                json.dumps(["signal"]),
                encoding="utf-8",
            )
            (seed / "decision.json").write_text(
                json.dumps({"threshold": 0.5}),
                encoding="utf-8",
            )
            (seed / "metrics.json").write_text("{}", encoding="utf-8")
            (run / "arguments.json").write_text(
                json.dumps({"model": "logistic", "scania_cost": False}),
                encoding="utf-8",
            )
            with (
                patch.object(api, "PROJECT_ROOT", root),
            ):
                response = asyncio.run(
                    self.request(
                        "POST",
                        "/v1/predict",
                        json={
                            "experiment": str(run),
                            "seed": 42,
                            "rows": [{"signal": 0.9}],
                        },
                    )
                )
            self.assertEqual(response.status_code, 200, response.text)
            result = response.json()
            self.assertEqual(result["n_predictions"], 1)
            self.assertEqual(result["predictions"][0]["prediction"], 1)
            self.assertEqual(len(result["predictions"][0]["scores"]), 2)

    def test_training_job_then_inference_end_to_end(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            processed = root / "datasets" / "processed"
            dataset = processed / "fixture"
            dataset.mkdir(parents=True)
            (dataset / "features.json").write_text(
                json.dumps(["signal"]),
                encoding="utf-8",
            )
            for split, values, labels in (
                ("train", [0.0, 0.1, 0.2, 0.3, 0.7, 0.8, 0.9, 1.0], [0, 0, 0, 0, 1, 1, 1, 1]),
                ("validation", [0.05, 0.25, 0.75, 0.95], [0, 0, 1, 1]),
                ("test", [0.02, 0.28, 0.72, 0.98], [0, 0, 1, 1]),
            ):
                pd.DataFrame(
                    {
                        "sample_id": [f"{split}_{index}" for index in range(len(values))],
                        "signal": values,
                        "target": labels,
                    }
                ).to_csv(dataset / f"{split}.csv", index=False)

            with (
                patch.object(api, "PROJECT_ROOT", root),
                patch.object(api, "API_PROCESSED", processed),
                patch.object(api, "API_RESULTS", root / "results" / "api"),
            ):
                response = asyncio.run(
                    self.request(
                        "POST",
                        "/v1/experiments",
                        json={
                            "dataset": str(dataset),
                            "model": "logistic",
                            "device": "cpu",
                            "seeds": [7],
                        },
                    )
                )
                self.assertEqual(response.status_code, 202, response.text)
                job_id = response.json()["job_id"]
                for _ in range(500):
                    job = asyncio.run(
                        self.request("GET", f"/v1/jobs/{job_id}")
                    ).json()
                    if job["status"] in {"completed", "failed"}:
                        break
                    time.sleep(0.02)
                self.assertEqual(job["status"], "completed", job.get("error"))
                output = job["result"]["output"]
                prediction = asyncio.run(
                    self.request(
                        "POST",
                        "/v1/predict",
                        json={
                            "experiment": output,
                            "seed": 7,
                            "rows": [{"signal": 0.95}],
                        },
                    )
                )
            self.assertEqual(prediction.status_code, 200, prediction.text)
            self.assertEqual(
                prediction.json()["predictions"][0]["prediction"],
                1,
            )


if __name__ == "__main__":
    unittest.main()
