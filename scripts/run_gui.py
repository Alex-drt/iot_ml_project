from __future__ import annotations

import csv
import json
import queue
import subprocess
import sys
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

try:
    import tkinter as tk
    from tkinter import (
        BOTH,
        END,
        LEFT,
        RIGHT,
        X,
        Y,
        filedialog,
        messagebox,
        simpledialog,
    )
    from tkinter import ttk
except ImportError:
    tk = None
    filedialog = None
    messagebox = None
    simpledialog = None
    ttk = None
    BOTH = END = LEFT = RIGHT = X = Y = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_PATH = PROJECT_ROOT / "iot_experiment.py.py"
ANALYZER_PATH = PROJECT_ROOT / "scripts" / "analyze_dataset.py"
SCANIA_PREPARER_PATH = PROJECT_ROOT / "scripts" / "prepare_scania.py"
CICIOMT_PREPARER_PATH = PROJECT_ROOT / "scripts" / "prepare_ciciomt.py"
REQUIRED_DATA_FILES = (
    "train.csv",
    "validation.csv",
    "test.csv",
    "features.json",
)
MODEL_LABELS = {
    "Logistic Regression (CPU)": "logistic",
    "Random Forest (CPU)": "random_forest",
    "MLP": "mlp",
    "LSTM autoencoder": "lstm_ae",
}
CPU_MODELS = {"logistic", "random_forest"}


@dataclass(frozen=True)
class PreparedDataset:
    directory: Path
    title: str
    task: str
    features: tuple[str, ...]
    headers: dict[str, tuple[str, ...]]
    preparation: dict


@dataclass(frozen=True)
class RunOptions:
    model: str
    device: str
    output: Path
    seeds: tuple[int, ...]
    target: str
    sample_id: str
    group: str = ""
    time: str = ""
    epochs: int = 20
    batch: int = 256
    hidden: int = 64
    latent: int = 16
    learning_rate: float = 0.001
    trees: int = 200
    jobs: int = 1
    window: int = 30
    quantile: float = 0.99
    max_gap: str = ""


def inspect_prepared_dataset(directory: Path) -> PreparedDataset:
    directory = Path(directory).expanduser().resolve()
    missing = [
        name
        for name in REQUIRED_DATA_FILES
        if not (directory / name).is_file()
    ]
    if missing:
        raise ValueError(
            f"В папке датасета отсутствуют файлы: {', '.join(missing)}"
        )

    try:
        features = json.loads(
            (directory / "features.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Не удалось прочитать features.json: {exc}") from exc

    if (
        not isinstance(features, list)
        or not features
        or any(not isinstance(name, str) or not name.strip() for name in features)
        or len({name.casefold() for name in features}) != len(features)
    ):
        raise ValueError(
            "features.json должен содержать упорядоченный список "
            "непустых уникальных имён признаков"
        )

    headers = {}
    for split in ("train", "validation", "test"):
        try:
            with (directory / f"{split}.csv").open(
                "r",
                encoding="utf-8-sig",
                newline="",
            ) as source:
                header = next(csv.reader(source), None)
        except (OSError, UnicodeError, csv.Error) as exc:
            raise ValueError(f"Не удалось прочитать {split}.csv: {exc}") from exc

        if not header:
            raise ValueError(f"{split}.csv пуст или не содержит заголовок")
        headers[split] = tuple(header)

    preparation_path = directory / "preparation.json"
    preparation = {}
    if preparation_path.is_file():
        try:
            preparation = json.loads(
                preparation_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"Не удалось прочитать preparation.json: {exc}"
            ) from exc
        if not isinstance(preparation, dict):
            raise ValueError("preparation.json должен содержать JSON-объект")

    title = preparation.get("dataset")
    if not isinstance(title, str) or not title.strip():
        title = directory.name

    task = preparation.get("task")
    if not isinstance(task, str) or not task.strip():
        task = "тип задачи не указан в preparation.json"

    return PreparedDataset(
        directory=directory,
        title=title,
        task=task,
        features=tuple(features),
        headers=headers,
        preparation=preparation,
    )


def discover_prepared_datasets(processed_root: Path) -> list[Path]:
    processed_root = Path(processed_root)
    if not processed_root.is_dir():
        return []

    return sorted(
        {
            candidate.parent.resolve()
            for candidate in processed_root.rglob("features.json")
            if all(
                (candidate.parent / name).is_file()
                for name in REQUIRED_DATA_FILES
            )
        },
        key=lambda path: str(path).casefold(),
    )


def build_scania_preparation_command(
    interpreter: str,
    preparer: Path,
    root: Path,
    output: Path,
    seed: int,
) -> list[str]:
    if not 0 <= seed < 2**32:
        raise ValueError("seed must satisfy 0 <= seed < 2**32")
    if output.exists():
        raise ValueError(
            f"Папка подготовленного датасета уже существует: {output}"
        )
    command = [
        interpreter,
        str(preparer),
        "--root",
        str(root),
        "--output",
        str(output),
        "--seed",
        str(seed),
    ]
    return command


def build_ciciomt_preparation_command(
    interpreter: str,
    preparer: Path,
    root: Path,
    manifest: Path,
    features: Path,
    output: Path,
) -> list[str]:
    if output.exists():
        raise ValueError(
            f"Папка подготовленного датасета уже существует: {output}"
        )
    for label, path in (
        ("Каталог исходных CSV", root),
        ("Манифест", manifest),
        ("Список признаков", features),
    ):
        if not path.exists():
            raise ValueError(f"{label} не найден: {path}")
    if not root.is_dir():
        raise ValueError(f"Каталог исходных CSV не является папкой: {root}")
    if not manifest.is_file() or not features.is_file():
        raise ValueError("Манифест и список признаков должны быть файлами")
    command = [
        interpreter,
        str(preparer),
        "--root",
        str(root),
        "--manifest",
        str(manifest),
        "--features",
        str(features),
        "--output",
        str(output),
    ]
    with manifest.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as stream:
        manifest_rows = csv.DictReader(stream)
        if any(
            row.get("split") == "validation"
            and row.get("source_partition") == "profiling"
            for row in manifest_rows
        ):
            command.extend(
                [
                    "--profiling-train-fraction",
                    "0.8",
                    "--seed",
                    "42",
                ]
            )
    return command


def build_prepared_output_path(
    dataset_name: str,
    now: datetime | None = None,
) -> Path:
    if dataset_name not in {"scania", "ciciomt"}:
        raise ValueError(f"Неизвестный набор данных: {dataset_name}")
    parent = PROJECT_ROOT / "datasets" / "processed" / dataset_name
    timestamp = (now or datetime.now()).strftime("%Y%m%d_%H%M%S")
    candidate = parent / f"prepared_{timestamp}"
    suffix = 2
    while candidate.exists():
        candidate = parent / f"prepared_{timestamp}_{suffix}"
        suffix += 1
    return candidate


def build_experiment_command(
    interpreter: str,
    experiment: Path,
    dataset: PreparedDataset,
    options: RunOptions,
    *,
    validate_only: bool = False,
) -> list[str]:
    if options.model not in MODEL_LABELS.values():
        raise ValueError(f"Неизвестная модель: {options.model}")
    if options.device not in {"cpu", "cuda"}:
        raise ValueError("Устройство должно быть cpu или cuda")
    if options.model in CPU_MODELS and options.device != "cpu":
        raise ValueError("Logistic Regression и Random Forest работают только на CPU")
    if options.model == "lstm_ae" and (not options.group or not options.time):
        raise ValueError("Для LSTM autoencoder укажите group и time")
    if options.output.exists() and not validate_only:
        raise ValueError(
            f"Папка результатов уже существует: {options.output}"
        )

    command = [
        interpreter,
        str(experiment),
        "--train",
        str(dataset.directory / "train.csv"),
        "--validation",
        str(dataset.directory / "validation.csv"),
        "--test",
        str(dataset.directory / "test.csv"),
        "--features",
        str(dataset.directory / "features.json"),
        "--model",
        options.model,
        "--device",
        options.device,
        "--target",
        options.target,
        "--id",
        options.sample_id,
        "--seeds",
        *(str(seed) for seed in options.seeds),
    ]

    if options.group:
        command.extend(["--group", options.group])
    if options.time:
        command.extend(["--time", options.time])
    if options.model in {"mlp", "lstm_ae"}:
        command.extend(
            [
                "--epochs",
                str(options.epochs),
                "--batch",
                str(options.batch),
                "--hidden",
                str(options.hidden),
                "--learning-rate",
                str(options.learning_rate),
            ]
        )
    if options.model == "random_forest":
        command.extend(
            ["--trees", str(options.trees), "--jobs", str(options.jobs)]
        )
    if options.model == "lstm_ae":
        command.extend(
            [
                "--latent",
                str(options.latent),
                "--window",
                str(options.window),
                "--quantile",
                str(options.quantile),
            ]
        )
        if options.max_gap:
            command.extend(["--max-gap", options.max_gap])
    if validate_only:
        command.append("--validate-only")
    else:
        command.extend(["--output", str(options.output)])

    return command


def environment_summary() -> tuple[str, bool]:
    details = [
        f"Python {sys.version.split()[0]}",
        str(Path(sys.executable).resolve()),
    ]
    try:
        import torch
    except ImportError:
        details.append("PyTorch не установлен; нейросети недоступны")
        return "\n".join(details), False

    details.append(f"PyTorch {torch.__version__}")
    details.append(f"CUDA runtime PyTorch: {torch.version.cuda or 'нет'}")
    try:
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            details.append(f"GPU: {torch.cuda.get_device_name(0)}")
        else:
            details.append("GPU: недоступен для этого PyTorch")
    except RuntimeError as exc:
        details.append(f"Проверка CUDA завершилась ошибкой: {exc}")
        return "\n".join(details), False
    return "\n".join(details), cuda_available


def resolve_device(model: str, selection: str, cuda_available: bool) -> str:
    if model in CPU_MODELS:
        return "cpu"
    devices = {"CPU": "cpu", "GPU (CUDA)": "cuda"}
    try:
        device = devices[selection]
    except KeyError as exc:
        raise ValueError(f"Неизвестный выбор устройства: {selection}") from exc
    if device == "cuda" and not cuda_available:
        raise ValueError("CUDA недоступна для выбранного Python-интерпретатора")
    return device


def build_run_output_path(parent: Path, run_name: str) -> Path:
    run_name = run_name.strip()
    if (
        not run_name
        or run_name in {".", ".."}
        or Path(run_name).name != run_name
        or Path(run_name).is_absolute()
    ):
        raise ValueError("Имя запуска должно быть одним именем папки")
    parent = Path(parent).expanduser().resolve()
    output = (parent / run_name).resolve()
    if output.parent != parent:
        raise ValueError("Имя запуска не должно выходить из выбранной папки")
    return output


def build_analysis_output_path(
    dataset: PreparedDataset,
    now: datetime | None = None,
) -> Path:
    dataset_name = dataset.directory.parent.name or dataset.directory.name
    parent = PROJECT_ROOT / "results" / "analytics" / dataset_name
    timestamp = (now or datetime.now()).strftime("%Y%m%d_%H%M%S")
    candidate = parent / f"analysis_{timestamp}"
    suffix = 2
    while candidate.exists():
        candidate = parent / f"analysis_{timestamp}_{suffix}"
        suffix += 1
    return candidate


def build_analysis_command(
    interpreter: str,
    analyzer: Path,
    dataset: PreparedDataset,
    output: Path,
    target: str,
) -> list[str]:
    if output.exists():
        raise ValueError(f"Папка аналитики уже существует: {output}")
    if not target or any(
        target not in dataset.headers[split]
        for split in ("train", "validation", "test")
    ):
        raise ValueError(f"Целевая колонка {target!r} должна быть во всех CSV")
    return [
        interpreter,
        str(analyzer),
        "--dataset",
        str(dataset.directory),
        "--output",
        str(output),
        "--target",
        target,
    ]


class ExperimentLauncher:
    def __init__(self, root):
        self.root = root
        self.root.title("IoT ML — запуск экспериментов")
        self.root.geometry("1020x860")
        self.root.minsize(860, 680)
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        self.messages = queue.Queue()
        self.dataset_choices: dict[str, Path] = {}
        self.dataset_var = tk.StringVar()
        self.dataset_path_var = tk.StringVar()
        self.output_parent_var = tk.StringVar(
            value=str(PROJECT_ROOT / "results")
        )
        self.run_name_var = tk.StringVar(
            value=f"run_{datetime.now():%Y%m%d_%H%M%S}"
        )
        self.model_var = tk.StringVar(
            value="MLP"
        )
        self.device_var = tk.StringVar(value="GPU (CUDA)")
        self.group_var = tk.StringVar(value="vehicle_id")
        self.time_var = tk.StringVar(value="time_step")
        self.target_var = tk.StringVar(value="target")
        self.id_var = tk.StringVar(value="sample_id")
        self.seeds_var = tk.StringVar(
            value="11 22 33"
        )
        self.epochs_var = tk.StringVar(value="20")
        self.batch_var = tk.StringVar(value="256")
        self.hidden_var = tk.StringVar(value="64")
        self.latent_var = tk.StringVar(value="16")
        self.learning_rate_var = tk.StringVar(
            value="0.001"
        )
        self.trees_var = tk.StringVar(value="200")
        self.jobs_var = tk.StringVar(value="1")
        self.window_var = tk.StringVar(value="30")
        self.quantile_var = tk.StringVar(value="0.99")
        self.max_gap_var = tk.StringVar()
        self.dataset_info_var = tk.StringVar(
            value="Выберите каталог подготовленного датасета."
        )
        self.environment_var = tk.StringVar()
        self.device_info_var = tk.StringVar()
        self.status_var = tk.StringVar(value="Готово")
        self.cuda_available = False
        self.busy = False

        self._build_layout()
        self._load_environment()
        self._refresh_dataset_choices()
        self.model_var.trace_add("write", self._update_model_controls)
        self.device_var.trace_add("write", self._update_device_description)
        self._update_model_controls()
        self._update_device_description()
        self.root.after(100, self._poll_messages)

    def _build_layout(self):
        outer = ttk.Frame(self.root, padding=12)
        outer.pack(fill=BOTH, expand=True)

        dataset_box = ttk.LabelFrame(
            outer,
            text="1. Подготовленный датасет",
            padding=10,
        )
        dataset_box.pack(fill=X, pady=(0, 8))

        self.dataset_combo = ttk.Combobox(
            dataset_box,
            textvariable=self.dataset_var,
            state="readonly",
        )
        self.dataset_combo.pack(side=LEFT, fill=X, expand=True)
        self.dataset_combo.bind(
            "<<ComboboxSelected>>",
            self._select_dataset_from_combo,
        )
        ttk.Button(
            dataset_box,
            text="Обзор...",
            command=self._browse_dataset,
        ).pack(side=LEFT, padx=(8, 0))
        prepare_box = ttk.Frame(dataset_box)
        prepare_box.pack(fill=X, pady=(8, 0))
        self.prepare_scania_button = ttk.Button(
            prepare_box,
            text="Подготовить SCANIA",
            command=self._prepare_scania,
        )
        self.prepare_scania_button.pack(side=LEFT)
        self.prepare_ciciomt_button = ttk.Button(
            prepare_box,
            text="Подготовить CICIoMT",
            command=self._prepare_ciciomt,
        )
        self.prepare_ciciomt_button.pack(side=LEFT, padx=(8, 0))
        ttk.Button(
            prepare_box,
            text="Обновить список",
            command=self._refresh_dataset_choices,
        ).pack(side=LEFT, padx=(8, 0))
        self.analysis_button = ttk.Button(
            prepare_box,
            text="Анализировать датасет",
            command=self._start_analysis,
        )
        self.analysis_button.pack(side=LEFT, padx=(8, 0))
        ttk.Label(
            dataset_box,
            textvariable=self.dataset_path_var,
            wraplength=900,
        ).pack(anchor="w", pady=(8, 0))
        ttk.Label(
            dataset_box,
            textvariable=self.dataset_info_var,
            justify=LEFT,
            wraplength=940,
        ).pack(anchor="w", pady=(5, 0))

        output_box = ttk.LabelFrame(
            outer,
            text="2. Куда сохранять результаты",
            padding=10,
        )
        output_box.pack(fill=X, pady=(0, 8))
        ttk.Label(output_box, text="Папка на компьютере:").grid(
            row=0, column=0, sticky="w"
        )
        ttk.Entry(
            output_box,
            textvariable=self.output_parent_var,
        ).grid(row=0, column=1, sticky="ew", padx=8)
        ttk.Button(
            output_box,
            text="Выбрать...",
            command=self._browse_output_parent,
        ).grid(row=0, column=2)
        ttk.Label(output_box, text="Имя нового запуска:").grid(
            row=1, column=0, sticky="w", pady=(8, 0)
        )
        ttk.Entry(
            output_box,
            textvariable=self.run_name_var,
        ).grid(row=1, column=1, sticky="ew", padx=8, pady=(8, 0))
        ttk.Label(
            output_box,
            text="Существующие папки не перезаписываются.",
        ).grid(row=1, column=2, sticky="w", pady=(8, 0))
        output_box.columnconfigure(1, weight=1)

        settings_row = ttk.Frame(outer)
        settings_row.pack(fill=X, pady=(0, 8))
        model_box = ttk.LabelFrame(
            settings_row,
            text="3. Модель и устройство вычислений",
            padding=10,
        )
        model_box.pack(side=LEFT, fill=BOTH, expand=True, padx=(0, 5))
        ttk.Label(model_box, text="Модель:").grid(row=0, column=0, sticky="w")
        self.model_combo = ttk.Combobox(
            model_box,
            textvariable=self.model_var,
            values=list(MODEL_LABELS),
            state="readonly",
        )
        self.model_combo.grid(row=0, column=1, sticky="ew", padx=8)
        ttk.Label(model_box, text="Вычисления:").grid(
            row=1, column=0, sticky="w", pady=(8, 0)
        )
        self.device_combo = ttk.Combobox(
            model_box,
            textvariable=self.device_var,
            values=("CPU", "GPU (CUDA)"),
            state="readonly",
        )
        self.device_combo.grid(
            row=1, column=1, sticky="ew", padx=8, pady=(8, 0)
        )
        ttk.Label(
            model_box,
            textvariable=self.device_info_var,
            wraplength=440,
        ).grid(row=2, column=0, columnspan=2, sticky="w", pady=(7, 0))
        model_box.columnconfigure(1, weight=1)

        environment_box = ttk.LabelFrame(
            settings_row,
            text="Текущее программно-аппаратное окружение",
            padding=10,
        )
        environment_box.pack(side=LEFT, fill=BOTH, expand=True, padx=(5, 0))
        ttk.Label(
            environment_box,
            textvariable=self.environment_var,
            justify=LEFT,
            wraplength=460,
        ).pack(anchor="w")

        params_box = ttk.LabelFrame(
            outer,
            text="4. Параметры модели",
            padding=10,
        )
        params_box.pack(fill=X, pady=(0, 8))
        fields = [
            ("Seeds (через пробел)", self.seeds_var),
            ("Эпохи", self.epochs_var),
            ("Batch size", self.batch_var),
            ("Hidden", self.hidden_var),
            ("Latent (LSTM-AE)", self.latent_var),
            ("Learning rate", self.learning_rate_var),
            ("Деревья (Random Forest)", self.trees_var),
            ("Jobs (Random Forest)", self.jobs_var),
            ("Window (LSTM-AE)", self.window_var),
            ("Quantile (LSTM-AE)", self.quantile_var),
            ("Max gap (необязательно)", self.max_gap_var),
        ]
        for index, (label, variable) in enumerate(fields):
            row, column = divmod(index, 4)
            cell = ttk.Frame(params_box)
            cell.grid(
                row=row,
                column=column,
                sticky="ew",
                padx=4,
                pady=3,
            )
            ttk.Label(cell, text=label).pack(anchor="w")
            ttk.Entry(cell, textvariable=variable, width=17).pack(
                fill=X
            )
        for column in range(4):
            params_box.columnconfigure(column, weight=1)

        meta_box = ttk.LabelFrame(
            outer,
            text="Метаданные CSV (не включаются в признаки)",
            padding=10,
        )
        meta_box.pack(fill=X, pady=(0, 8))
        metadata = [
            ("ID строки", self.id_var),
            ("Целевая колонка", self.target_var),
            ("Группа (LSTM)", self.group_var),
            ("Время (LSTM)", self.time_var),
        ]
        for index, (label, variable) in enumerate(metadata):
            ttk.Label(meta_box, text=label).grid(
                row=0, column=index * 2, sticky="w"
            )
            ttk.Entry(meta_box, textvariable=variable, width=20).grid(
                row=0, column=index * 2 + 1, sticky="ew", padx=(5, 10)
            )
        for column in (1, 3, 5, 7):
            meta_box.columnconfigure(column, weight=1)

        action_row = ttk.Frame(outer)
        action_row.pack(fill=X, pady=(0, 6))
        self.run_button = ttk.Button(
            action_row,
            text="Проверить данные и запустить модель",
            command=self._start_run,
        )
        self.run_button.pack(side=LEFT)
        ttk.Label(
            action_row,
            textvariable=self.status_var,
        ).pack(side=LEFT, padx=12)

        log_box = ttk.LabelFrame(
            outer,
            text="Проверка, ход запуска и сообщения",
            padding=6,
        )
        log_box.pack(fill=BOTH, expand=True)
        self.log = tk.Text(
            log_box,
            height=12,
            wrap="word",
            state="disabled",
        )
        scrollbar = ttk.Scrollbar(
            log_box,
            orient="vertical",
            command=self.log.yview,
        )
        self.log.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=RIGHT, fill=Y)
        self.log.pack(side=LEFT, fill=BOTH, expand=True)

    def _load_environment(self):
        summary, self.cuda_available = environment_summary()
        self.environment_var.set(summary)

    def _refresh_dataset_choices(self):
        selected_path = self.dataset_path_var.get().strip()
        self.dataset_choices.clear()
        for directory in discover_prepared_datasets(
            PROJECT_ROOT / "datasets" / "processed"
        ):
            try:
                dataset = inspect_prepared_dataset(directory)
            except ValueError:
                continue
            label = f"{dataset.title} — {directory}"
            self.dataset_choices[label] = directory
        self.dataset_combo.configure(values=list(self.dataset_choices))
        if self.dataset_choices:
            selected = next(
                (
                    label
                    for label, directory in self.dataset_choices.items()
                    if str(directory) == selected_path
                ),
                next(iter(self.dataset_choices)),
            )
            self.dataset_var.set(selected)
            self._select_dataset(self.dataset_choices[selected])
        else:
            self.dataset_var.set("")
            self.dataset_path_var.set("")
            self.dataset_info_var.set(
                "В datasets/processed пока нет подготовленных наборов. "
                "Нажмите «Обзор...» и укажите папку с train.csv, "
                "validation.csv, test.csv и features.json."
            )

    def _set_busy(self, busy: bool, status: str):
        self.busy = busy
        state = "disabled" if busy else "normal"
        self.run_button.configure(state=state)
        self.prepare_scania_button.configure(state=state)
        self.prepare_ciciomt_button.configure(state=state)
        self.analysis_button.configure(state=state)
        self.status_var.set(status)

    def _start_preparation(self, command: list[str], output: Path, label: str):
        if output.exists():
            messagebox.showerror(
                "Папка уже существует",
                f"Выберите новый каталог для подготовки:\n{output}",
            )
            return
        try:
            output.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            messagebox.showerror(
                "Не удалось создать каталог",
                f"Не удалось подготовить папку {output.parent}: {exc}",
            )
            return
        confirm = messagebox.askyesno(
            "Подтвердите подготовку",
            f"Набор: {label}\n"
            f"Результат: {output}\n\n"
            "Подготовка прочитает исходные файлы и создаст новую папку. "
            "Продолжить?",
        )
        if not confirm:
            return
        self._set_busy(True, "Подготовка датасета…")
        self._append_log(f"Подготовка {label}; результат: {output}")
        threading.Thread(
            target=self._preparation_worker,
            args=(command, output, label),
            daemon=True,
        ).start()

    def _prepare_scania(self):
        root = PROJECT_ROOT / "datasets" / "raw" / "scania_component_x"
        if not root.is_dir():
            root = filedialog.askdirectory(
                title="Выберите каталог исходных CSV SCANIA",
                initialdir=str(PROJECT_ROOT / "datasets" / "raw"),
            )
            if not root:
                return
            root = Path(root)
        seed = simpledialog.askinteger(
            "Подготовка SCANIA",
            "Seed выборки:",
            initialvalue=42,
            minvalue=0,
            maxvalue=2**32 - 1,
        )
        if seed is None:
            return
        output = build_prepared_output_path("scania")
        try:
            command = build_scania_preparation_command(
                sys.executable,
                SCANIA_PREPARER_PATH,
                Path(root),
                output,
                seed,
            )
        except ValueError as exc:
            messagebox.showerror("Нельзя подготовить SCANIA", str(exc))
            return
        self._start_preparation(command, output, "SCANIA Component X")

    def _prepare_ciciomt(self):
        default_root = (
            PROJECT_ROOT
            / "datasets"
            / "raw"
            / "ciciomt2024"
            / "CICIoMT2024"
            / "WIFI_and_MQTT"
            / "attacks"
            / "csv"
        )
        root = filedialog.askdirectory(
            title="Выберите CICIoMT CSV с папками train/test",
            initialdir=str(default_root if default_root.is_dir() else PROJECT_ROOT),
        )
        if not root:
            return
        manifest = filedialog.askopenfilename(
            title="Выберите проверенный CICIoMT манифест",
            initialdir=str(PROJECT_ROOT / "configs"),
            filetypes=(("CSV", "*.csv"), ("Все файлы", "*.*")),
        )
        if not manifest:
            return
        features = filedialog.askopenfilename(
            title="Выберите проверенный список признаков",
            initialdir=str(PROJECT_ROOT / "configs"),
            filetypes=(("JSON", "*.json"), ("Все файлы", "*.*")),
        )
        if not features:
            return
        output = build_prepared_output_path("ciciomt")
        try:
            command = build_ciciomt_preparation_command(
                sys.executable,
                CICIOMT_PREPARER_PATH,
                Path(root),
                Path(manifest),
                Path(features),
                output,
            )
        except ValueError as exc:
            messagebox.showerror("Нельзя подготовить CICIoMT", str(exc))
            return
        self._start_preparation(command, output, "CICIoMT2024")

    def _preparation_worker(
        self,
        command: list[str],
        output: Path,
        label: str,
    ):
        try:
            result = self._run_process(command)
            if result != 0:
                self.messages.put(
                    (
                        "prepared",
                        (
                            f"Подготовка {label} завершилась с кодом {result}.",
                            False,
                            None,
                        ),
                    )
                )
                return
            inspect_prepared_dataset(output)
            self.messages.put(
                (
                    "prepared",
                    (
                        f"Подготовлен набор {label}: {output}",
                        True,
                        output,
                    ),
                )
            )
        except (OSError, RuntimeError, ValueError) as exc:
            self.messages.put(
                (
                    "prepared",
                    (
                        f"Ошибка подготовки {label}: "
                        f"{type(exc).__name__}: {exc}",
                        False,
                        None,
                    ),
                )
            )

    def _select_dataset_from_combo(self, _event=None):
        selection = (
            _event.widget.get()
            if _event is not None
            else self.dataset_var.get()
        )
        directory = self.dataset_choices.get(selection)
        if directory is not None:
            self._select_dataset(directory)

    def _select_dataset(self, directory: Path):
        self.dataset_path_var.set(str(directory))
        try:
            dataset = inspect_prepared_dataset(directory)
        except ValueError as exc:
            self.dataset_info_var.set(f"Ошибка датасета: {exc}")
            self._append_log(f"Ошибка датасета: {exc}")
            return

        rows = []
        for split, header in dataset.headers.items():
            path = directory / f"{split}.csv"
            size_mb = path.stat().st_size / (1024 * 1024)
            rows.append(
                f"{split}.csv: {len(header)} колонок, {size_mb:.2f} MiB"
            )
        self.dataset_info_var.set(
            f"Набор: {dataset.title}\n"
            f"Задача: {dataset.task}\n"
            f"Признаков: {len(dataset.features)} "
            f"({', '.join(dataset.features[:8])}"
            f"{'…' if len(dataset.features) > 8 else ''})\n"
            + " | ".join(rows)
        )

        header_set = set(dataset.headers["train"])
        if "vehicle_id" in header_set:
            self.group_var.set("vehicle_id")
        elif "sequence_id" in header_set:
            self.group_var.set("sequence_id")
        if "time_step" in header_set:
            self.time_var.set("time_step")
        if "target" in header_set:
            self.target_var.set("target")
        if "sample_id" in header_set:
            self.id_var.set("sample_id")

    def _browse_dataset(self):
        selected = filedialog.askdirectory(
            title="Выберите папку с prepared train/validation/test"
        )
        if not selected:
            return
        directory = Path(selected)
        try:
            dataset = inspect_prepared_dataset(directory)
        except ValueError as exc:
            messagebox.showerror("Некорректный датасет", str(exc))
            return
        label = f"{dataset.title} — {directory}"
        self.dataset_choices[label] = dataset.directory
        self.dataset_combo.configure(values=list(self.dataset_choices))
        self.dataset_var.set(label)
        self._select_dataset(dataset.directory)

    def _browse_output_parent(self):
        selected = filedialog.askdirectory(
            title="Выберите папку, где создать каталог результатов"
        )
        if selected:
            self.output_parent_var.set(selected)

    def _start_analysis(self):
        dataset_path = self.dataset_path_var.get().strip()
        if not dataset_path:
            messagebox.showerror(
                "Нельзя выполнить анализ",
                "Сначала выберите подготовленный датасет.",
            )
            return
        try:
            dataset = inspect_prepared_dataset(Path(dataset_path))
            output = build_analysis_output_path(dataset)
            command = build_analysis_command(
                sys.executable,
                ANALYZER_PATH,
                dataset,
                output,
                self.target_var.get().strip(),
            )
        except (OSError, ValueError) as exc:
            messagebox.showerror("Нельзя выполнить анализ", str(exc))
            return
        if not messagebox.askyesno(
            "Подтвердите анализ",
            f"Датасет: {dataset.title}\n"
            f"Аналитика будет сохранена в:\n{output}\n\n"
            "Исходные CSV не изменяются. Продолжить?",
        ):
            return

        self._set_busy(True, "Анализ датасета…")
        self._append_log(
            f"Анализ датасета: {dataset.title}\n"
            f"Результаты: {output}"
        )
        threading.Thread(
            target=self._analysis_worker,
            args=(command, output),
            daemon=True,
        ).start()

    def _analysis_worker(self, command: list[str], output: Path):
        try:
            result = self._run_process(command)
            if result == 0:
                self.messages.put(
                    ("done", (f"Анализ завершён: {output}", True))
                )
            else:
                self.messages.put(
                    (
                        "done",
                        (
                            f"Анализ завершился с кодом {result}. "
                            f"Проверьте каталог {output}.",
                            False,
                        ),
                    )
                )
        except (OSError, RuntimeError, ValueError) as exc:
            self.messages.put(
                (
                    "done",
                    (f"Ошибка анализа: {type(exc).__name__}: {exc}", False),
                )
            )

    def _model_key(self) -> str:
        return MODEL_LABELS.get(self.model_var.get(), "mlp")

    def _update_model_controls(self, *_args):
        if _args and _args[-1] != "write":
            return
        model = self._model_key()
        if model in CPU_MODELS:
            self.device_var.set("CPU")
            self.device_combo.configure(state="disabled")
        else:
            self.device_combo.configure(state="readonly")
            if self.device_var.get() == "GPU (CUDA)" and not self.cuda_available:
                self.device_var.set("CPU")
        self._update_device_description()

    def _update_device_description(self, *_args):
        if _args and _args[-1] != "write":
            return
        model = self._model_key()
        if model in CPU_MODELS:
            self.device_info_var.set(
                "Эта модель выполняется только на CPU."
            )
        elif self.device_var.get() == "GPU (CUDA)":
            self.device_info_var.set(
                "Будет запрошена CUDA. Наличие GPU/версия появятся "
                "в метриках запуска; при недоступности запуск завершится "
                "ошибкой."
                if self.cuda_available
                else "CUDA недоступна в текущем Python-окружении."
            )
        else:
            self.device_info_var.set(
                "Нейросеть будет запущена на CPU."
            )

    def _parse_options(self) -> tuple[PreparedDataset, RunOptions]:
        dataset_path = self.dataset_path_var.get().strip()
        if not dataset_path:
            raise ValueError("Сначала выберите подготовленный датасет")
        dataset = inspect_prepared_dataset(Path(dataset_path))
        model = self._model_key()
        device = resolve_device(
            model,
            self.device_var.get(),
            self.cuda_available,
        )
        if model == "lstm_ae":
            if not self.group_var.get().strip() or not self.time_var.get().strip():
                raise ValueError(
                    "Для LSTM укажите колонку группы и настоящую колонку "
                    "времени. Время нельзя заменять номером строки."
                )
            for split, header in dataset.headers.items():
                for column in (self.group_var.get().strip(), self.time_var.get().strip()):
                    if column not in header:
                        raise ValueError(
                            f"В {split}.csv нет колонки {column!r}"
                        )

        try:
            seeds = tuple(
                int(value)
                for value in self.seeds_var.get().replace(",", " ").split()
            )
        except ValueError as exc:
            raise ValueError("Seeds должны быть целыми числами") from exc
        if not seeds or len(set(seeds)) != len(seeds):
            raise ValueError("Укажите уникальные seed, например: 11 22 33")
        if any(seed < 0 or seed >= 2**32 for seed in seeds):
            raise ValueError("Каждый seed должен быть в диапазоне [0, 2**32)")

        try:
            output_parent = Path(self.output_parent_var.get()).expanduser()
            output = build_run_output_path(
                output_parent,
                self.run_name_var.get(),
            )
            numeric = {
                "epochs": int(self.epochs_var.get()),
                "batch": int(self.batch_var.get()),
                "hidden": int(self.hidden_var.get()),
                "latent": int(self.latent_var.get()),
                "learning_rate": float(self.learning_rate_var.get()),
                "trees": int(self.trees_var.get()),
                "jobs": int(self.jobs_var.get()),
                "window": int(self.window_var.get()),
                "quantile": float(self.quantile_var.get()),
            }
        except (ValueError, OSError) as exc:
            raise ValueError(f"Проверьте параметры: {exc}") from exc

        if not self.target_var.get().strip() or not self.id_var.get().strip():
            raise ValueError("Укажите имена целевой колонки и ID")

        options = RunOptions(
            model=model,
            device=device,
            output=output,
            seeds=seeds,
            target=self.target_var.get().strip(),
            sample_id=self.id_var.get().strip(),
            group=self.group_var.get().strip(),
            time=self.time_var.get().strip(),
            max_gap=self.max_gap_var.get().strip(),
            **numeric,
        )
        build_experiment_command(
            sys.executable,
            EXPERIMENT_PATH,
            dataset,
            options,
        )
        return dataset, options

    def _start_run(self):
        try:
            dataset, options = self._parse_options()
        except (OSError, ValueError) as exc:
            messagebox.showerror("Нельзя запустить эксперимент", str(exc))
            return

        if options.model in {"mlp", "lstm_ae"} and options.device == "cuda":
            device_label = f"GPU: {self._gpu_name()}"
        else:
            device_label = "CPU"
        confirm = messagebox.askyesno(
            "Подтвердите запуск",
            f"Датасет: {dataset.title}\n"
            f"Модель: {options.model}\n"
            f"Вычисления: {device_label}\n"
            f"Seed: {', '.join(map(str, options.seeds))}\n"
            f"Результаты: {options.output}\n\n"
            "Будет выполнена проверка входных данных, затем обучение. "
            "Входные CSV не изменяются. Продолжить?",
        )
        if not confirm:
            return

        self._set_busy(True, "Проверка данных…")
        self._append_log(
            f"Датасет: {dataset.title} ({dataset.directory})\n"
            f"Модель: {options.model}; устройство: {device_label}\n"
            f"Результаты будут сохранены в: {options.output}"
        )
        threading.Thread(
            target=self._run_worker,
            args=(dataset, options),
            daemon=True,
        ).start()

    def _gpu_name(self) -> str:
        try:
            import torch

            return torch.cuda.get_device_name(0)
        except (ImportError, RuntimeError) as exc:
            return f"CUDA недоступна: {exc}"

    def _run_worker(self, dataset: PreparedDataset, options: RunOptions):
        try:
            validation_command = build_experiment_command(
                sys.executable,
                EXPERIMENT_PATH,
                dataset,
                options,
                validate_only=True,
            )
            self.messages.put(("status", "Проверка схемы и split…"))
            validation = self._run_process(validation_command)
            if validation != 0:
                self.messages.put(
                    ("done", ("Проверка данных не пройдена", False))
                )
                return

            self.messages.put(("status", "Обучение модели…"))
            command = build_experiment_command(
                sys.executable,
                EXPERIMENT_PATH,
                dataset,
                options,
            )
            result = self._run_process(command)
            if result == 0:
                self.messages.put(
                    ("done", (f"Запуск завершён: {options.output}", True))
                )
            else:
                self.messages.put(
                    (
                        "done",
                        (
                            f"CLI завершился с кодом {result}. "
                            f"Проверьте status.json в {options.output}, "
                            "если каталог уже создан.",
                            False,
                        ),
                    )
                )
        except (OSError, RuntimeError, ValueError) as exc:
            self.messages.put(
                (
                    "done",
                    (f"Ошибка запуска: {type(exc).__name__}: {exc}", False),
                )
            )

    def _run_process(self, command: list[str]) -> int:
        self.messages.put(("log", "$ " + subprocess.list2cmdline(command)))
        try:
            process = subprocess.Popen(
                command,
                cwd=PROJECT_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
        except OSError as exc:
            self.messages.put(("log", f"Ошибка запуска процесса: {exc}"))
            return 1

        if process.stdout is None:
            raise RuntimeError("Не удалось открыть вывод процесса")
        for line in process.stdout:
            self.messages.put(("log", line.rstrip()))
        return process.wait()

    def _poll_messages(self):
        try:
            while True:
                kind, payload = self.messages.get_nowait()
                if kind == "log":
                    self._append_log(payload)
                elif kind == "status":
                    self.status_var.set(payload)
                elif kind == "done":
                    text, success = payload
                    self._set_busy(
                        False,
                        "Завершено" if success else "Ошибка",
                    )
                    self._append_log(text)
                    if success:
                        messagebox.showinfo("Готово", text)
                    else:
                        messagebox.showerror("Ошибка запуска", text)
                elif kind == "prepared":
                    text, success, directory = payload
                    self._set_busy(
                        False,
                        "Подготовка завершена" if success else "Ошибка",
                    )
                    self._append_log(text)
                    if success and directory is not None:
                        self.dataset_path_var.set(str(directory))
                        self._refresh_dataset_choices()
                    if success:
                        messagebox.showinfo("Готово", text)
                    else:
                        messagebox.showerror("Ошибка подготовки", text)
        except queue.Empty:
            pass
        self.root.after(100, self._poll_messages)

    def _append_log(self, text: str):
        self.log.configure(state="normal")
        self.log.insert(END, text + "\n")
        if float(self.log.index("end-1c").split(".")[0]) > 1200:
            self.log.delete("1.0", "200.0")
        self.log.see(END)
        self.log.configure(state="disabled")

    def close(self):
        if self.busy:
            messagebox.showwarning(
                "Запуск выполняется",
                "Дождитесь завершения проверки/обучения, прежде чем закрывать окно.",
            )
            return
        self.root.destroy()


def main():
    if tk is None or ttk is None:
        raise RuntimeError(
            "Tkinter is required for the GUI launcher. Install the Python "
            "Tk bindings for your interpreter and run again."
        )
    root = tk.Tk()
    ExperimentLauncher(root)
    root.mainloop()


if __name__ == "__main__":
    main()
