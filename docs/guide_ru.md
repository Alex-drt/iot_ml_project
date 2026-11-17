# Руководство по запуску проекта IoT ML (Windows)

Все команды ниже предназначены для PowerShell и используют единое окружение
`.venv`, чтобы не зависеть от случайно выбранного системного Python.

## 1. Краткий setup

Выполните из корня проекта. Python 3.15 должен быть установлен и доступен
через Windows Python Launcher `py`.

```powershell
cd D:\iot_ml_project
if (-not (Test-Path .venv\Scripts\python.exe)) { py -3.15 -m venv .venv }
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements\base.txt
.\.venv\Scripts\python.exe -m pip install -r requirements\api.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py" -q
```

Для нейросетей с CUDA установите PyTorch в это же окружение (не требуется
для Logistic Regression и Random Forest):

```powershell
.\.venv\Scripts\python.exe -m pip install --index-url https://download.pytorch.org/whl/cu132 --extra-index-url https://pypi.org/simple "torch==2.15.0+cu132"
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

Если `torch.cuda.is_available()` равен `False`, используйте `--device cpu`
или проверьте драйвер, GPU и установленный PyTorch wheel.

## 2. REST API

Запустите сервер локально:

```powershell
.\.venv\Scripts\python.exe -m uvicorn scripts.api:app --host 127.0.0.1 --port 8000
```

Откройте http://127.0.0.1:8000/docs для интерактивной документации.
Долгие операции возвращают `job_id`; проверяйте состояние через
`GET /v1/jobs/{job_id}`. Поддерживаются:

- `POST /v1/validate` — проверка датасета;
- `POST /v1/experiments` — обучение;
- `POST /v1/analysis` — аналитика и графики;
- `POST /v1/prepare/scania` и `POST /v1/prepare/ciciomt` — подготовка;
- `POST /v1/predict` — inference по сохранённому запуску.

Датасеты для train/validate/analyze должны находиться внутри
`datasets/processed`; исходные данные и конфиги подготовки ограничены
каталогами `datasets/raw` и `configs`. Результаты создаются в `results/api`
или `datasets/processed/<dataset>/api_*`. Prediction принимает JSON `rows`
с признаками модели; для LSTM-AE нужны group/time и полные окна, результат
содержит конечные точки окон. Ограничение запроса — 10 000 строк.

API по умолчанию слушает только `127.0.0.1`. Не открывайте его наружу:
аутентификация и разграничение пользователей не реализованы. Загружайте
только собственные доверенные model artifacts. Список задач хранится в
памяти процесса и очищается при его перезапуске.

## 3. Подготовка данных

Для CICIoMT и SCANIA скрипты подготовки создают новые папки в `datasets/processed` и не перезаписывают исходники.

Пример:

```powershell
.\.venv\Scripts\python.exe scripts\prepare_scania.py --root datasets\raw\scania_component_x --output datasets\processed\scania\prepared_v1 --seed 42
```

Для CICIoMT:

```powershell
.\.venv\Scripts\python.exe scripts\prepare_ciciomt.py --root datasets\raw\ciciomt2024 --manifest configs\ciciomt_manifest.csv --features configs\ciciomt_features.json --output datasets\processed\ciciomt\prepared_v1 --seed 42
```

## 4. Анализ подготовленного датасета

```powershell
.\.venv\Scripts\python.exe scripts\analyze_dataset.py --dataset datasets\processed\scania\prepared_v1 --output results\analytics\scania\analysis_v1
```

После этого появятся:
- `analysis.json`;
- PNG-графики по распределению классов, пропускам, feature distributions и корреляциям.

## 5. Запуск обучения

Основной CLI:

```powershell
.\.venv\Scripts\python.exe iot_experiment.py.py --train datasets\processed\scania\prepared_v1\train.csv --validation datasets\processed\scania\prepared_v1\validation.csv --test datasets\processed\scania\prepared_v1\test.csv --features datasets\processed\scania\prepared_v1\features.json --model logistic --device cpu --output results\scania\run_001
```

Поддерживаемые модели:
- `logistic`
- `random_forest`
- `mlp`
- `lstm_ae`

### Пример для MLP

```powershell
.\.venv\Scripts\python.exe iot_experiment.py.py --train datasets\processed\scania\prepared_v1\train.csv --validation datasets\processed\scania\prepared_v1\validation.csv --test datasets\processed\scania\prepared_v1\test.csv --features datasets\processed\scania\prepared_v1\features.json --model mlp --device cpu --epochs 20 --batch 128 --hidden 64 --learning-rate 0.001 --output results\scania\mlp_run_001
```

## 6. GUI

```powershell
.\.venv\Scripts\python.exe scripts\run_gui.py
```

GUI позволяет:
- выбрать подготовленный датасет;
- запустить подготовку CICIoMT/SCANIA;
- выбрать модель и параметры;
- запустить обучение и анализ результатов.

## 7. Ограничения проекта

- Это исследовательский прототип, не production-система; метрики не
  гарантируют качество в эксплуатации.
- Все CSV split загружаются в память. На очень больших датасетах нужны
  достаточная RAM и место для результатов.
- В SCANIA классы сильно несбалансированы; readout'ы одного автомобиля
  коррелируют, поэтому число строк не равно числу независимых объектов.
- В CICIoMT метки capture-level, validation на profiling может отличаться
  по домену от train/test, а между split есть повторяющиеся feature-векторы.
- У CICIoMT нет подтверждённой временной семантики порядка строк. Не
  используйте LSTM-AE без проверенных group/time и последовательностей.
- Анализатор строит описательные отчёты и графики; он не заменяет оценку на
  отложенной выборке.
- GUI требует Tkinter. Нейросетевые модели требуют PyTorch; CUDA зависит
  также от совместимого драйвера и доступного GPU.

Подробное описание dataset-specific ограничений, результатов и методики
см. в [docs/project_report_ru.md](project_report_ru.md).

## 8. Полезные правила

- Всегда используйте один интерпретатор: `.\.venv\Scripts\python.exe`.
- Не перезаписывайте `results/*` и `datasets/processed/*` без необходимости.
- Сохраняйте новый результат в отдельную папку.
- Для проверки качества не используйте только descriptive analysis — нужен held-out evaluation.
