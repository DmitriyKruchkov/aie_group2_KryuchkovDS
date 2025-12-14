# S03 – eda_cli: мини-EDA для CSV

Небольшое CLI-приложение для базового анализа CSV-файлов.
Используется в рамках Семинара 03 курса «Инженерия ИИ».

## Требования

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) установлен в систему

## Инициализация проекта

В корне проекта (S03):

```bash
uv sync
```

Эта команда:

- создаст виртуальное окружение `.venv`;
- установит зависимости из `pyproject.toml`;
- установит сам проект `eda-cli` в окружение.

## Запуск CLI

### Краткий обзор

```bash
uv run eda-cli overview data/example.csv
```

Параметры:

- `--sep` – разделитель (по умолчанию `,`);
- `--encoding` – кодировка (по умолчанию `utf-8`).

### Полный EDA-отчёт

```bash
uv run eda-cli report data/example.csv --out-dir reports
```

Дополнительные параметры команды `report`:

- `--max-hist-columns` – максимум числовых колонок для гистограмм (по умолчанию `6`);
- `--top-k-categories` – количество top-значений для категориальных признаков (по умолчанию `5`);
- `--title` – заголовок отчёта (по умолчанию `"EDA-отчёт"`);
- `--min-missing-share` – порог доли пропусков для выделения проблемных колонок (по умолчанию `0.1` или 10%).

Пример с новыми параметрами:

```bash
uv run eda-cli report data/example.csv --out-dir reports --max-hist-columns 8 --top-k-categories 10 --title "Анализ пользовательских данных" --min-missing-share 0.05
```

В результате в каталоге `reports/` появятся:

- `report.md` – основной отчёт в Markdown;
- `summary.csv` – таблица по колонкам;
- `missing.csv` – пропуски по колонкам;
- `correlation.csv` – корреляционная матрица (если есть числовые признаки);
- `top_categories/*.csv` – top-k категорий по строковым признакам;
- `hist_*.png` – гистограммы числовых колонок;
- `missing_matrix.png` – визуализация пропусков;
- `correlation_heatmap.png` – тепловая карта корреляций.

## Тесты

```bash
uv run pytest -q
```

## HTTP-сервис (FastAPI)

Проект включает HTTP-сервис для анализа качества датасетов через REST API.

### Запуск сервиса

```bash
uv run uvicorn eda_cli.api:app --reload --port 8000
```

После запуска сервис будет доступен по адресу `http://localhost:8000`.

Интерактивная документация API доступна по адресу:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### Эндпоинты

#### `GET /health`

Проверка работоспособности сервиса.

**Пример запроса:**
```bash
curl http://localhost:8000/health
```

**Ответ:**
```json
{
  "status": "ok"
}
```

#### `POST /quality`

Оценка качества данных на основе входных параметров (без загрузки файла).

**Пример запроса:**
```bash
curl -X POST http://localhost:8000/quality \
  -H "Content-Type: application/json" \
  -d '{
    "n_rows": 1000,
    "n_cols": 10,
    "max_missing_share": 0.1,
    "too_few_rows": false,
    "too_many_columns": false
  }'
```

**Ответ:**
```json
{
  "ok_for_model": true,
  "quality_score": 0.9,
  "latency_ms": 0.123,
  "flags": {
    "too_few_rows": false,
    "too_many_columns": false,
    "too_many_missing": false,
    "max_missing_share": 0.1
  }
}
```

#### `POST /quality-from-csv`

Оценка качества данных из загруженного CSV-файла.

**Пример запроса:**
```bash
curl -X POST http://localhost:8000/quality-from-csv \
  -F "file=@data/example.csv"
```

**Ответ:**
```json
{
  "ok_for_model": true,
  "quality_score": 0.85,
  "latency_ms": 45.67,
  "flags": {
    "too_few_rows": false,
    "too_many_columns": false,
    "too_many_missing": false,
    "max_missing_share": 0.05
  }
}
```

#### `POST /quality-flags-from-csv`

Полный набор флагов качества данных из загруженного CSV-файла. Использует все эвристики качества, включая новые из HW03:
- `has_constant_columns` — наличие константных колонок
- `has_high_cardinality_categoricals` — высокая кардинальность категориальных признаков
- `has_suspicious_id_duplicates` — подозрительные дубликаты в ID-колонках
- `has_many_zero_values` — большое количество нулей в числовых колонках

**Пример запроса:**
```bash
curl -X POST http://localhost:8000/quality-flags-from-csv \
  -F "file=@data/example.csv"
```

**Ответ:**
```json
{
  "flags": {
    "too_few_rows": false,
    "too_many_columns": false,
    "too_many_missing": false,
    "has_constant_columns": false,
    "has_high_cardinality_categoricals": true,
    "has_suspicious_id_duplicates": false,
    "has_many_zero_values": true
  }
}
```

### Использование через Python

```python
import requests

# Проверка здоровья сервиса
response = requests.get("http://localhost:8000/health")
print(response.json())

# Анализ CSV-файла
with open("data/example.csv", "rb") as f:
    files = {"file": f}
    response = requests.post(
        "http://localhost:8000/quality-from-csv",
        files=files
    )
    print(response.json())
```
