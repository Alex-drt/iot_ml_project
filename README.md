# IoT ML project

Проект для подготовки данных, проверки пайплайна, обучения моделей и оценки результатов на датасетах CICIoMT и SCANIA Component X.

Ключевые возможности:
- CLI для обучения: logistic, random forest, MLP, LSTM autoencoder;
- подготовка данных для CICIoMT и SCANIA;
- анализ prepared splits и построение графиков;
- GUI для запуска подготовки и экспериментов;
- регрессионные тесты на временных fixtures.

Основные документы:
- [docs/guide_ru.md](docs/guide_ru.md) — короткое руководство по запуску;
- [docs/project_report_ru.md](docs/project_report_ru.md) — результаты проекта и аналитика.

## Структура проекта

```text
.
├── iot_experiment.py.py      # основной CLI
├── scripts/
│   ├── prepare_ciciomt.py    # подготовка CICIoMT
│   ├── prepare_scania.py     # подготовка SCANIA
│   ├── analyze_dataset.py    # анализ prepared splits
│   ├── api.py                # локальный REST API
│   └── run_gui.py            # Tkinter GUI
├── configs/                  # манифесты, признаки, конфиги
├── datasets/
│   ├── archives/             # исходные архивы
│   ├── raw/                  # распакованные исходники
│   └── processed/            # подготовленные train/validation/test
├── results/                  # эксперименты и аналитика
├── requirements/
│   ├── base.txt              # базовые зависимости
│   └── environment-lock.txt  # snapshot базовых зависимостей
├── tests/                    # регрессионные тесты
├── docs/                     # документация
├── README.md
├── .venv                     # единое рабочее окружение
└── .gitignore
```

## Windows-only: быстрый запуск

Откройте PowerShell в корне проекта. При первом запуске создайте `.venv`;
затем используйте только его интерпретатор:

```powershell
cd D:\iot_ml_project
if (-not (Test-Path .venv\Scripts\python.exe)) { py -3.15 -m venv .venv }
.\.venv\Scripts\python.exe -m pip install -r requirements\base.txt
.\.venv\Scripts\python.exe -m pip install -r requirements\api.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py" -q
```

Для нейросетей с CUDA установите PyTorch в ту же среду:

```powershell
.\.venv\Scripts\python.exe -m pip install --index-url https://download.pytorch.org/whl/cu132 --extra-index-url https://pypi.org/simple "torch==2.15.0+cu132"
```

Запустите локальный REST API:

```powershell
.\.venv\Scripts\python.exe -m uvicorn scripts.api:app --host 127.0.0.1 --port 8000
```

Документация API: http://127.0.0.1:8000/docs. Сервер привязан к localhost;
не публикуйте его в сеть без аутентификации и мер безопасности.

Замените `prepared_v1` и `quick_run` на существующий каталог подготовленного
набора и новое имя запуска:

```powershell
.\.venv\Scripts\python.exe iot_experiment.py.py --train datasets\processed\scania\prepared_v1\train.csv --validation datasets\processed\scania\prepared_v1\validation.csv --test datasets\processed\scania\prepared_v1\test.csv --features datasets\processed\scania\prepared_v1\features.json --model logistic --device cpu --output results\scania\quick_run
```

Для подробных команд подготовки данных и GUI см. [docs/guide_ru.md](docs/guide_ru.md).

## Требования к данным

Для корректной работы эксперимент принимает:
- 3 CSV: train, validation, test;
- один JSON `features.json` со списком признаков;
- непустые уникальные `sample_id`;
- столбец целевой метки `target`;
- все признаки из `features.json` должны присутствовать в каждом split.

## Важные правила

- Не трогайте исходные данные в `datasets/raw` и `datasets/archives`.
- Для каждого подготовленного набора или запуска создавайте новую папку.
- Не включайте `target`, `sample_id`, `group`, `time` в список признаков.
- Проверяйте, что классы в обучении и валидации корректно разделены и contiguously labeled.
- GUI запускается только при наличии Tkinter в текущем Python.

## Ограничения

- Это исследовательский прототип, не production-система и не гарантия
  качества прогноза в эксплуатации.
- CSV split целиком загружаются в память; очень большие наборы могут
  требовать значительной RAM.
- SCANIA сильно несбалансирован: редкие классы имеют мало наблюдений, а
  readout'ы одного автомобиля коррелируют.
- CICIoMT метки задаются на уровне capture; profiling validation может
  отличаться по домену от train/test. Между split также встречаются
  повторяющиеся feature-векторы.
- Для CICIoMT не подтверждена временная интерпретация порядка строк;
  LSTM-AE применим только к последовательностям с обоснованными group/time.
- Анализ датасета описательный и не заменяет проверку на независимой
  отложенной выборке.

## Примечания по окружению

В проекте используется одно рабочее окружение `.venv` для всех задач. Это упрощает запуск, исключает расхождения между версиями Python/PyTorch и снижает вероятность «магического» сбоя на машине разработчика или пользователя.

Текущая проверенная конфигурация:
- Python 3.15
- PyTorch 2.15.0+cu132
- CUDA runtime 13.2
- unit tests проходят успешно
