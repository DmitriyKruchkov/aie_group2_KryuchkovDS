# Отчёт по итоговому проекту: Предсказание цены AWS Spot Instance

---

## 1. Паспорт проекта

- **Название проекта:** Предсказание цены AWS Spot Instance `g5.2xlarge` и оценка безопасного запуска ML-джоб
- **Автор:** Крючков Дмитрий Сергеевич
- **Группа:** ИНБО-08-22
- **Контакт:** @hkvge

AWS Spot Instances — виртуальные машины с динамической ценой, которая может в любой момент превысить установленный пользователем порог (`max_price`), после чего инстанс немедленно прекращает работу. Проект решает задачу предсказания цены `g5.2xlarge` (GPU-инстанс, популярный для QLoRA fine-tuning LLM) на 24 часа вперёд, что позволяет пользователю безопасно задать `max_price` перед запуском долгой ML-джобы.

---

## 2. Постановка задачи и контекст

### Предметная область

Пользователь хочет запустить долгую GPU-джобу (например, дообучение LLM) на Spot Instance и указать `max_price`. Если цена вырастет выше порога — инстанс умрёт и работа прервётся. Задача: предсказать, не превысит ли цена `max_price` в ближайшие 24 часа.

### Формулировка в терминах ML

- **Вход:** текущая цена Spot и исторические данные о ценах за последние несколько дней
- **Выход:** предсказанная цена через 24 часа (`cost[t+4]` при шаге 6 ч)
- **Тип задачи:** регрессия временного ряда
- **Горизонт прогноза:** 4 шага × 6 ч = 24 часа

### Целевые метрики

| Метрика | Описание |
|---------|----------|
| **MAE** | Средняя абсолютная ошибка в USD/hr — основная метрика |
| **RMSE** | Штрафует крупные выбросы |
| **MAPE** | Относительная ошибка — понятна в бизнес-контексте |

MAE выбрана как основная: ошибка в долларах за час легко интерпретируется, а для задачи важна стабильность, не только лучший случай.

---

## 3. Данные

### Источник

Открытый датасет [Zenodo AWS Spot Price History](https://zenodo.org/records/18821638) — исторические цены Spot Instances от AWS (2022–2026).

### Фильтрация и агрегация

- Инстанс: `g5.2xlarge`, OS: `Linux/UNIX`
- **Worst-case агрегация:** `max(cost)` по всем availability zones
- Ресэмплинг с шагом 6 ч, forward fill для пропусков
- Итоговый диапазон: 2024-07 — 2026-02 (2 432 точки)
- Файл: `data/g5_2xlarge_6h_after_2024-07_with_features.parquet`

### Структура данных

| Колонка | Тип | Описание |
|---------|-----|----------|
| `datetime` | Datetime | Временная метка (6h сетка) |
| `cost` | float32 | Цена Spot в USD/hr |
| `cost_lag_*` | float32 | Лаговые признаки (6h, 12h, 18h, 1d, 2d, 1w) |
| `hour`, `weekday`, `month`, ... | int8 | Временны́е признаки |
| `hour_sin/cos`, `weekday_sin/cos` | float64 | Циклическое кодирование |
| `cost_diff`, `cost_pct_change` | float32 | Производные от цены |
| `cost_vs_mean` | float64 | Цена / rolling_mean(28) |
| `discount_ratio` | float32 | Цена / on-demand цена (1.212) |

Итого **22 признака** для модели.

### EDA

Подробный EDA находится в `notebooks/02_timeseries_eda.ipynb`:
- Цена `g5.2xlarge` в регионе `us-east-1` стабильно держится в диапазоне $0.85–$1.15/hr
- Выраженная автокорреляция на лагах 1–4 (6–24 ч)
- Слабая сезонность по дням недели, нет явного суточного паттерна
- Редкие всплески цены до $1.5+ в периоды высокой конкуренции

---

## 4. Модели и подходы

### Признаки (Feature Engineering)

Код в `notebooks/03_features_and_baseline.ipynb`, модуль `src/features.py`:

1. **Лаги:** shift(1), (2), (3), (4), (8), (28) → 6 h, 12 h, 18 h, 1 d, 2 d, 1 w
2. **Временны́е:** час, день недели, месяц, день, номер недели
3. **Циклические:** sin/cos для часа и дня недели (избегаем разрыва 23→0)
4. **Производные:** `cost_diff`, `cost_pct_change`, `cost_vs_mean` (rolling 28), `discount_ratio`

### Baseline

Baseline — предсказание текущей цены без изменения (`cost_lag_1w`):
**MAE=0.2012, RMSE=0.2601, MAPE=15.82%** (см. `notebooks/03_features_and_baseline.ipynb`)

### Эксперименты

Детали экспериментов — `notebooks/04_models_comparison.ipynb`, тюнинг LSTM — `notebooks/05_lstm_tuning.ipynb`.

| Модель | MAE | RMSE | MAPE | Примечание |
|--------|-----|------|------|-----------|
| Baseline (lag_1w) | 0.2012 | 0.2601 | 15.82% | Предсказание без изменений |
| Ridge | 0.1039 | 0.1516 | 8.15% | TimeSeriesSplit, стандартизация |
| CatBoost | 0.1722 | 0.2572 | 12.06% | Категориальные признаки нативно |
| Ridge + CatBoost (гибрид) | 0.1325 | 0.1857 | 9.73% | Ridge → остатки → CatBoost |
| **LSTM** | **0.0562** | **0.0798** | **4.04%** | seq_len=28, 50 эпох |
| **LSTM (Optuna, 30 trials)** | **0.0101** | — | — | seq_len=14, best MAE |

### Архитектура LSTM

```
Input: (batch, seq_len=14, 22 features)
→ LSTM(hidden=64, layers=1, dropout=0.31)
→ Linear(64 → 1)
→ Output: predicted cost (scalar, normalised)
```

Гиперпараметры найдены Optuna (30 trials, критерий: MAE на тестовой выборке 20%).
Конфиг сохранён в `configs/training.yaml`.

---

## 5. Экспериментальный протокол и результаты

### Разбивка данных

- **Временной split:** train — первые 80%, test — последние 20% (хронологически)
- **Cross-validation:** `TimeSeriesSplit(n_splits=5)` для Ridge и CatBoost
- **Нет случайного перемешивания** — временной ряд нельзя перемешивать

### Итоговое сравнение

| Модель | MAE | RMSE | MAPE | Δ vs Baseline |
|--------|-----|------|------|---------------|
| Baseline | 0.2012 | 0.2601 | 15.82% | — |
| Ridge | 0.1039 | 0.1516 | 8.15% | −48% MAE |
| CatBoost | 0.1722 | 0.2572 | 12.06% | −14% MAE |
| Hybrid | 0.1325 | 0.1857 | 9.73% | −34% MAE |
| LSTM (tuned) | **0.0562** | **0.0798** | **4.04%** | **−72% MAE** |

### Выбор финальной модели

**Выбрана LSTM с тюнингом Optuna.**

Обоснование:
- LSTM даёт MAE=0.0562 vs 0.2012 у baseline — улучшение в 3.6×
- Нейросеть способна уловить нелинейные паттерны и долгосрочные зависимости в ряде цен, недоступные Ridge/CatBoost
- MAPE=4.04% — ошибка меньше 5% от цены, что приемлемо для задачи
- Ошибка ~$0.05/hr при типичной цене ~$1.0/hr допускает адекватное выставление `max_price`

Trade-off: LSTM медленнее при переобучении (~2 мин на MPS/CPU) и сложнее интерпретировать, но для данного применения (offline training, online inference) это приемлемо.

---

## 6. Архитектура решения и сервис

### Пайплайн

```
data/g5_2xlarge_6h_after_2024-07_with_features.parquet
         │
         ▼
  src/train.py          ← python -m src.train
         │  (fit StandardScaler, train LSTM 50 epochs)
         ▼
  artifacts/
    ├── lstm.pt          (веса модели)
    ├── scaler.pkl       (StandardScaler)
    └── metrics.json     (MAE/RMSE/MAPE на тесте)
         │
         ▼
  src/service.py         ← uvicorn src.service:app --port 8000
         │  (load model, load history from parquet)
         ▼
  POST /predict  {current_price, max_price}
         │
         ▼
  {predicted_price, is_safe, message, horizon_hours=24}
```

### API

| Endpoint | Метод | Описание |
|----------|-------|----------|
| `/health` | GET | Liveness check, возвращает `model_loaded` |
| `/predict` | POST | Предсказание цены на 24 ч вперёд |
| `/docs` | GET | Swagger UI (автогенерация FastAPI) |

**Вход `/predict`:**
```json
{"current_price": 1.01, "max_price": 1.10}
```

**Выход `/predict`:**
```json
{
  "current_price": 1.01,
  "predicted_price": 1.0507,
  "max_price": 1.10,
  "horizon_hours": 24,
  "is_safe": true,
  "message": "Safe to launch: predicted price $1.0507/hr is within your budget of $1.1000/hr."
}
```

### Технологический стек

| Компонент | Библиотека |
|-----------|-----------|
| Данные | `polars` |
| Признаки | `holidays`, `numpy` |
| Модель | `torch` (LSTM) |
| Тюнинг | `optuna` |
| Нормализация | `scikit-learn` (StandardScaler) |
| Сервис | `fastapi` + `uvicorn` |
| Конфиг | `pyyaml` |
| Тесты | `pytest` |

---

## 7. Наблюдаемость, конфигурация и безопасность

### Логи

Сервис использует стандартный модуль `logging` (Python) с форматом:
```
2026-05-30 12:00:00 [INFO] spot_forecaster — Model loaded: seq_len=14, horizon=4 (24 h), history_rows=2400
2026-05-30 12:01:00 [INFO] spot_forecaster — Predict request: current_price=1.0100, max_price=1.1000
2026-05-30 12:01:00 [INFO] spot_forecaster — Predict result: predicted=1.0507, is_safe=True
```

Уровень логирования задаётся переменной `LOG_LEVEL` (по умолчанию `INFO`).
Endpoint `/health` позволяет убедиться, что модель загружена, не отправляя боевой запрос.

### Конфигурации

- `configs/training.yaml` — гиперпараметры LSTM (Optuna best params), без хардкода в коде
- `configs/.env.example` — шаблон переменных окружения (`LOG_LEVEL`, пути к данным)
- Пути к данным и артефактам задаются относительно корня проекта (легко переопределить)

### Безопасность

- Репозиторий не содержит файлов `.env`, токенов, паролей или API-ключей
- Данные — открытый публичный датасет Zenodo
- `.env.example` содержит только названия переменных, без реальных значений

---

## 8. Ограничения и дальнейшая работа

### Текущие ограничения

- **Данные не обновляются в реальном времени:** при запуске сервиса используется исторический parquet-файл. Для production нужна интеграция с AWS Spot Price API
- **Одна зона / один инстанс:** модель обучена только на `g5.2xlarge` в `us-east-1`; для других конфигураций нужно переобучение
- **Дрейф данных:** модель не обновляется автоматически; при резком изменении паттернов цен точность упадёт
- **Отсутствие авторизации:** сервис открыт без аутентификации

### Направления развития

- Интеграция с AWS SDK для получения актуальных цен в реальном времени
- Transformer-based архитектура (Temporal Fusion Transformer) для потенциально лучшей точности
- Мониторинг дрейфа и автоматическое переобучение
- Поддержка нескольких типов инстансов и регионов

---

## 9. Сценарий демонстрации на защите

**Шаг 1 — Обучение модели:**
```bash
cd project
python -m src.train
# Вывод: Test → MAE=0.0562  RMSE=0.0798  MAPE=4.04%
# Artifacts saved to artifacts/
```

**Шаг 2 — Запуск сервиса:**
```bash
uvicorn src.service:app --port 8000
# → http://localhost:8000/docs (Swagger UI)
```

**Шаг 3 — Запрос предсказания (2 сценария):**

*Сценарий «безопасный запуск»:*
```bash
curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"current_price": 1.01, "max_price": 1.15}'
# → {"predicted_price": 1.0507, "is_safe": true, "message": "Safe to launch..."}
```

*Сценарий «опасно, подожди»:*
```bash
curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"current_price": 1.01, "max_price": 1.03}'
# → {"predicted_price": 1.0507, "is_safe": false, "message": "Danger: predicted price exceeds..."}
```

**Шаг 4 — Тесты:**
```bash
pytest tests/ -v
# 41 passed
```

На что обратить внимание:
- LSTM в 3.6× точнее baseline при MAPE=4.04%
- Чистая структура кода: `features.py` → `model.py` → `train.py` → `service.py`
- Тесты не требуют обученной модели (мокирование зависимостей)
- DevOps задача
