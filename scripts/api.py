from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_PATH = PROJECT_ROOT / "iot_experiment.py.py"
ANALYZER_PATH = PROJECT_ROOT / "scripts" / "analyze_dataset.py"
SCANIA_PREPARER_PATH = PROJECT_ROOT / "scripts" / "prepare_scania.py"
CICIOMT_PREPARER_PATH = PROJECT_ROOT / "scripts" / "prepare_ciciomt.py"
API_RESULTS = PROJECT_ROOT / "results" / "api"
API_PROCESSED = PROJECT_ROOT / "datasets" / "processed"
MAX_PREDICTION_ROWS = 10000

_spec = importlib.util.spec_from_file_location(
    "iot_experiment",
    EXPERIMENT_PATH,
)
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"Unable to load experiment module: {EXPERIMENT_PATH}")
EXPERIMENT = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = EXPERIMENT
_spec.loader.exec_module(EXPERIMENT)

app = FastAPI(
    title="IoT ML API",
    version="1.0.0",
    description=(
        "Local REST API for dataset validation, preparation, analysis, "
        "model training, and inference."
    ),
)
_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="iot-api")
_jobs: dict[str, dict[str, Any]] = {}
_jobs_lock = threading.Lock()


class DatasetRequest(BaseModel):
    dataset: str
    model: Literal["logistic", "random_forest", "mlp", "lstm_ae"] = "logistic"
    device: Literal["cpu", "cuda", "auto"] = "cpu"
    target: str = "target"
    sample_id: str = "sample_id"
    group: str | None = None
    time: str | None = None
    seeds: list[int] = Field(default_factory=lambda: [42], min_length=1)
    epochs: int = Field(default=20, ge=1)
    batch: int = Field(default=256, ge=1)
    hidden: int = Field(default=64, ge=1)
    latent: int = Field(default=16, ge=1)
    window: int = Field(default=30, ge=2)
    trees: int = Field(default=200, ge=1)
    jobs: int = Field(default=1)
    learning_rate: float = Field(default=0.001, gt=0)
    quantile: float = Field(default=0.99, gt=0, lt=1)
    threshold_metric: Literal["balanced_accuracy", "f1"] = "balanced_accuracy"
    max_gap: float | None = Field(default=None, gt=0)
    scania_cost: bool = False
    endpoint_only_evaluation: bool = False
    max_train_windows_per_group: int | None = Field(default=None, ge=1)


class AnalysisRequest(BaseModel):
    dataset: str
    target: str = "target"
    sample_size: int = Field(default=10000, ge=1)
    chunksize: int = Field(default=50000, ge=1)


class ScaniaPreparationRequest(BaseModel):
    root: str
    seed: int = Field(default=42, ge=0, lt=2**32)
    training_sampling: Literal[
        "one_per_vehicle",
        "all_eligible_readouts",
    ] = "all_eligible_readouts"
    write_anomaly_sequences: bool = False


class CiciomtPreparationRequest(BaseModel):
    root: str
    manifest: str
    features: str
    seed: int = Field(default=42, ge=0, lt=2**32)
    chunksize: int = Field(default=50000, ge=1)
    profiling_train_fraction: float = Field(default=0, ge=0, lt=1)


class PredictionRequest(BaseModel):
    experiment: str
    seed: int = Field(default=42, ge=0, lt=2**32)
    rows: list[dict[str, Any]] = Field(min_length=1, max_length=MAX_PREDICTION_ROWS)


def _resolve_inside(value: str, root: Path, label: str) -> Path:
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root.resolve(strict=True))
    except (OSError, ValueError) as exc:
        raise HTTPException(
            status_code=400,
            detail=f"{label} must resolve to a path inside {root}",
        ) from exc
    return resolved


def _dataset_path(value: str) -> Path:
    path = _resolve_inside(value, API_PROCESSED, "dataset")
    if not path.is_dir():
        raise HTTPException(status_code=400, detail="dataset must be a directory")
    return path


def _command_for_dataset(
    dataset: Path,
    request: DatasetRequest,
    *,
    validate_only: bool,
) -> list[str]:
    if len(set(request.seeds)) != len(request.seeds):
        raise HTTPException(status_code=422, detail="seeds must be distinct")
    if any(seed < 0 or seed >= 2**32 for seed in request.seeds):
        raise HTTPException(status_code=422, detail="seeds must be in [0, 2**32)")
    if request.jobs == 0:
        raise HTTPException(status_code=422, detail="jobs cannot be 0")
    if request.model == "lstm_ae" and (not request.group or not request.time):
        raise HTTPException(
            status_code=422,
            detail="lstm_ae requires group and time columns",
        )
    if request.model != "lstm_ae" and (
        request.max_gap is not None
        or request.endpoint_only_evaluation
        or request.max_train_windows_per_group is not None
    ):
        raise HTTPException(
            status_code=422,
            detail="window-specific options require model='lstm_ae'",
        )

    command = [
        sys.executable,
        str(EXPERIMENT_PATH),
        "--train",
        str(dataset / "train.csv"),
        "--validation",
        str(dataset / "validation.csv"),
        "--test",
        str(dataset / "test.csv"),
        "--features",
        str(dataset / "features.json"),
        "--model",
        request.model,
        "--device",
        request.device,
        "--target",
        request.target,
        "--id",
        request.sample_id,
        "--seeds",
        *(str(seed) for seed in request.seeds),
        "--threshold-metric",
        request.threshold_metric,
    ]
    for argument, value in (("--group", request.group), ("--time", request.time)):
        if value:
            command.extend([argument, value])
    if request.model in {"mlp", "lstm_ae"}:
        command.extend(
            [
                "--epochs",
                str(request.epochs),
                "--batch",
                str(request.batch),
                "--hidden",
                str(request.hidden),
                "--learning-rate",
                str(request.learning_rate),
            ]
        )
    if request.model == "random_forest":
        command.extend(["--trees", str(request.trees), "--jobs", str(request.jobs)])
    if request.model == "lstm_ae":
        command.extend(
            [
                "--latent",
                str(request.latent),
                "--window",
                str(request.window),
                "--quantile",
                str(request.quantile),
            ]
        )
        if request.max_gap is not None:
            command.extend(["--max-gap", str(request.max_gap)])
        if request.endpoint_only_evaluation:
            command.append("--endpoint-only-evaluation")
        if request.max_train_windows_per_group is not None:
            command.extend(
                [
                    "--max-train-windows-per-group",
                    str(request.max_train_windows_per_group),
                ]
            )
    if request.scania_cost:
        command.append("--scania-cost")

    if validate_only:
        command.append("--validate-only")
    else:
        output = _new_output(API_RESULTS, "experiment")
        command.extend(["--output", str(output)])
    return command


def _new_output(parent: Path, prefix: str) -> Path:
    parent.mkdir(parents=True, exist_ok=True)
    return parent / f"{prefix}_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{uuid.uuid4().hex[:8]}"


def _run_command(command: list[str]) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=24 * 60 * 60,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("Command exceeded the 24-hour execution limit") from exc
    if completed.returncode:
        detail = completed.stderr[-8000:] or completed.stdout[-8000:]
        raise RuntimeError(
            f"Command failed with exit code {completed.returncode}: {detail}"
        )
    return {
        "returncode": completed.returncode,
        "stdout": completed.stdout[-8000:],
    }


def _submit_job(kind: str, operation) -> dict[str, Any]:
    job_id = uuid.uuid4().hex
    created = datetime.now(timezone.utc).isoformat()
    with _jobs_lock:
        _jobs[job_id] = {
            "id": job_id,
            "kind": kind,
            "status": "queued",
            "created_at": created,
            "finished_at": None,
            "result": None,
            "error": None,
        }
    _executor.submit(_run_job, job_id, operation)
    return {"job_id": job_id, "status": "queued"}


def _run_job(job_id: str, operation) -> None:
    with _jobs_lock:
        _jobs[job_id]["status"] = "running"
    try:
        result = operation()
    except Exception as exc:
        with _jobs_lock:
            _jobs[job_id].update(
                status="failed",
                finished_at=datetime.now(timezone.utc).isoformat(),
                error=f"{type(exc).__name__}: {exc}",
            )
        return
    with _jobs_lock:
        _jobs[job_id].update(
            status="completed",
            finished_at=datetime.now(timezone.utc).isoformat(),
            result=result,
        )


def _job_snapshot(job_id: str) -> dict[str, Any]:
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job not found")
        return dict(job)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/validate", status_code=202)
def validate_dataset(request: DatasetRequest) -> dict[str, str]:
    dataset = _dataset_path(request.dataset)
    command = _command_for_dataset(dataset, request, validate_only=True)
    return _submit_job(
        "validation",
        lambda: _run_command(command),
    )


@app.post("/v1/experiments", status_code=202)
def train_model(request: DatasetRequest) -> dict[str, str]:
    dataset = _dataset_path(request.dataset)
    command = _command_for_dataset(dataset, request, validate_only=False)
    output = Path(command[command.index("--output") + 1])
    return _submit_job(
        "training",
        lambda: {
            **_run_command(command),
            "output": str(output),
        },
    )


@app.get("/v1/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    return _job_snapshot(job_id)


@app.post("/v1/analysis", status_code=202)
def analyze(request: AnalysisRequest) -> dict[str, str]:
    dataset = _dataset_path(request.dataset)
    output = _new_output(API_RESULTS, "analysis")
    command = [
        sys.executable,
        str(ANALYZER_PATH),
        "--dataset",
        str(dataset),
        "--output",
        str(output),
        "--target",
        request.target,
        "--sample-size",
        str(request.sample_size),
        "--chunksize",
        str(request.chunksize),
    ]
    return _submit_job(
        "analysis",
        lambda: {**_run_command(command), "output": str(output)},
    )


@app.post("/v1/prepare/scania", status_code=202)
def prepare_scania(request: ScaniaPreparationRequest) -> dict[str, str]:
    raw_root = PROJECT_ROOT / "datasets" / "raw"
    root = _resolve_inside(request.root, raw_root, "root")
    if not root.is_dir():
        raise HTTPException(status_code=400, detail="root must be a directory")
    output = _new_output(API_PROCESSED / "scania", "api")
    command = [
        sys.executable,
        str(SCANIA_PREPARER_PATH),
        "--root",
        str(root),
        "--output",
        str(output),
        "--seed",
        str(request.seed),
        "--training-sampling",
        request.training_sampling,
    ]
    if request.write_anomaly_sequences:
        command.append("--write-anomaly-sequences")
    return _submit_job(
        "prepare_scania",
        lambda: {**_run_command(command), "output": str(output)},
    )


@app.post("/v1/prepare/ciciomt", status_code=202)
def prepare_ciciomt(request: CiciomtPreparationRequest) -> dict[str, str]:
    root = _resolve_inside(
        request.root,
        PROJECT_ROOT / "datasets" / "raw",
        "root",
    )
    manifest = _resolve_inside(
        request.manifest,
        PROJECT_ROOT / "configs",
        "manifest",
    )
    features = _resolve_inside(
        request.features,
        PROJECT_ROOT / "configs",
        "features",
    )
    if not root.is_dir() or not manifest.is_file() or not features.is_file():
        raise HTTPException(
            status_code=400,
            detail="root must be a directory and manifest/features must be files",
        )
    output = _new_output(API_PROCESSED / "ciciomt", "api")
    command = [
        sys.executable,
        str(CICIOMT_PREPARER_PATH),
        "--root",
        str(root),
        "--manifest",
        str(manifest),
        "--features",
        str(features),
        "--output",
        str(output),
        "--seed",
        str(request.seed),
        "--chunksize",
        str(request.chunksize),
        "--profiling-train-fraction",
        str(request.profiling_train_fraction),
    ]
    return _submit_job(
        "prepare_ciciomt",
        lambda: {**_run_command(command), "output": str(output)},
    )


def _torch_model(model_name: str, n_features: int, n_classes: int, args: dict):
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise HTTPException(
            status_code=503,
            detail="PyTorch is required to load this model",
        ) from exc

    if model_name == "mlp":
        return nn.Sequential(
            nn.Linear(n_features, int(args["hidden"])),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(int(args["hidden"]), int(args["hidden"])),
            nn.ReLU(),
            nn.Linear(int(args["hidden"]), n_classes),
        )
    if model_name == "lstm_ae":
        class Autoencoder(nn.Module):
            def __init__(self):
                super().__init__()
                hidden = int(args["hidden"])
                latent = int(args["latent"])
                self.encoder = nn.LSTM(n_features, hidden, batch_first=True)
                self.to_z = nn.Linear(hidden, latent)
                self.from_z = nn.Linear(latent, hidden)
                self.decoder = nn.LSTM(hidden, hidden, batch_first=True)
                self.output = nn.Linear(hidden, n_features)

            def forward(self, inputs):
                _, (hidden_state, _) = self.encoder(inputs)
                latent_state = self.to_z(hidden_state[-1])
                repeated = (
                    self.from_z(latent_state)
                    .unsqueeze(1)
                    .expand(-1, inputs.shape[1], -1)
                )
                decoded, _ = self.decoder(repeated)
                return self.output(decoded)

        return Autoencoder()
    raise HTTPException(status_code=400, detail=f"Unsupported model: {model_name}")


def predict(request: PredictionRequest) -> dict[str, Any]:
    run_root = _resolve_inside(request.experiment, PROJECT_ROOT / "results", "experiment")
    seed_dir = _resolve_inside(
        str(run_root / f"seed_{request.seed}"),
        run_root,
        "seed",
    )
    required = [
        seed_dir / "preprocessing.joblib",
        seed_dir / "features.json",
        seed_dir / "decision.json",
    ]
    if not all(path.is_file() for path in required):
        raise HTTPException(
            status_code=404,
            detail="Experiment seed is incomplete or does not exist",
        )
    try:
        args = json.loads((run_root / "arguments.json").read_text(encoding="utf-8"))
        features = json.loads((seed_dir / "features.json").read_text(encoding="utf-8"))
        decision = json.loads((seed_dir / "decision.json").read_text(encoding="utf-8"))
        metrics = json.loads((seed_dir / "metrics.json").read_text(encoding="utf-8"))
        model_name = str(args["model"])
        frame = pd.DataFrame(request.rows)
        missing = [feature for feature in features if feature not in frame.columns]
        if missing:
            raise HTTPException(status_code=422, detail=f"Missing features: {missing}")
        values = frame[features].apply(pd.to_numeric, errors="raise")
        values = values.replace([np.inf, -np.inf], np.nan).to_numpy(dtype=np.float64)
        preprocessing = joblib.load(seed_dir / "preprocessing.joblib")
        transformed = preprocessing.transform(values)
        EXPERIMENT.require_finite("prediction preprocessing", transformed)
        transformed = transformed.astype(np.float32)
        EXPERIMENT.require_finite("prediction float32 conversion", transformed)
    except HTTPException:
        raise
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=422, detail=f"Invalid prediction input or artifact: {exc}") from exc

    if model_name in {"logistic", "random_forest"}:
        model_path = seed_dir / "model.joblib"
        if not model_path.is_file():
            raise HTTPException(status_code=404, detail="Model artifact not found")
        model = joblib.load(model_path)
        scores = model.predict_proba(transformed)
        predicted = (
            EXPERIMENT.scania_cost_decision(scores)
            if args.get("scania_cost")
            else (
                (scores[:, 1] >= decision["threshold"]).astype(np.int64)
                if len(model.classes_) == 2
                else np.argmax(scores, axis=1)
            )
        )
        EXPERIMENT.require_finite("prediction scores", scores)
        endpoints = np.arange(len(frame))
    else:
        model_path = seed_dir / "model.pt"
        if not model_path.is_file():
            raise HTTPException(status_code=404, detail="Model artifact not found")
        try:
            import torch

            checkpoint = torch.load(
                model_path,
                map_location="cpu",
                weights_only=True,
            )
            n_classes = len(metrics["per_class"])
            model = _torch_model(
                model_name,
                len(features),
                n_classes,
                args,
            )
            model.load_state_dict(checkpoint["state_dict"])
            model.eval()
            if model_name == "mlp":
                with torch.inference_mode():
                    scores = torch.softmax(
                        model(torch.from_numpy(transformed)),
                        dim=1,
                    ).numpy()
                EXPERIMENT.require_finite("prediction scores", scores)
                endpoints = np.arange(len(frame))
                predicted = (
                    (scores[:, 1] >= decision["threshold"]).astype(np.int64)
                    if n_classes == 2
                    else (
                        EXPERIMENT.scania_cost_decision(scores)
                        if args.get("scania_cost")
                        else np.argmax(scores, axis=1)
                    )
                )
            else:
                group_column = args.get("group")
                time_column = args.get("time")
                if (
                    not group_column
                    or not time_column
                    or group_column not in frame
                    or time_column not in frame
                ):
                    raise HTTPException(
                        status_code=422,
                        detail="LSTM-AE prediction requires group and time columns",
                    )
                frame[time_column] = pd.to_numeric(
                    frame[time_column],
                    errors="raise",
                )
                frame[group_column] = frame[group_column].astype("string")
                if frame[group_column].isna().any():
                    raise HTTPException(
                        status_code=422,
                        detail="group values cannot be missing",
                    )
                window_args = argparse.Namespace(
                    group=group_column,
                    time=time_column,
                    window=int(args["window"]),
                    max_gap=args.get("max_gap"),
                    endpoint_only_evaluation=True,
                )
                specification = EXPERIMENT.build_window_index(
                    frame,
                    np.zeros(len(frame), dtype=np.int64),
                    window_args,
                )
                windows = EXPERIMENT.LazyWindowView(
                    transformed,
                    specification,
                    int(args["window"]),
                )
                batches = []
                with torch.inference_mode():
                    for start in range(0, len(windows), int(args["batch"])):
                        batch = torch.from_numpy(
                            np.stack(
                                [
                                    windows[index]
                                    for index in range(
                                        start,
                                        min(start + int(args["batch"]), len(windows)),
                                    )
                                ]
                            )
                        )
                        batches.append(
                            ((model(batch) - batch) ** 2)
                            .mean(dim=(1, 2))
                            .numpy()
                        )
                scores = np.concatenate(batches)
                EXPERIMENT.require_finite("prediction scores", scores)
                predicted = (scores > decision["threshold"]).astype(np.int64)
                endpoints = specification["endpoints"]
        except HTTPException:
            raise
        except (ImportError, OSError, KeyError, TypeError, ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=422, detail=f"Unable to run model inference: {exc}") from exc

    score_values: list[Any]
    if model_name == "lstm_ae":
        score_values = [float(value) for value in scores]
        predictions = [
            {"row_index": int(row), "score": float(score), "prediction": int(label)}
            for row, score, label in zip(endpoints, score_values, predicted)
        ]
    else:
        score_values = scores.tolist()
        predictions = [
            {
                "row_index": int(row),
                "scores": [float(value) for value in score],
                "prediction": int(label),
            }
            for row, score, label in zip(endpoints, score_values, predicted)
        ]
    return {
        "model": model_name,
        "seed": request.seed,
        "n_predictions": len(predictions),
        "predictions": predictions,
    }


app.post("/v1/predict")(predict)


def main() -> None:
    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError(
            "Install API dependencies with: python -m pip install -r requirements\\api.txt"
        ) from exc
    uvicorn.run(
        "scripts.api:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
    )


if __name__ == "__main__":
    main()
