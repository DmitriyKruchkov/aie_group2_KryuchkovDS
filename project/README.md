# Итоговый проект: Предсказание цены AWS Spot Instance

---

## 1. Паспорт проекта

- **Название проекта:** Предсказание цены AWS Spot Instance `g5.2xlarge` и оценка безопасного запуска ML-джоб
- **Автор:** Крючков Дмитрий Сергеевич
- **Группа:** ИНБО-08-22
- **Контакт:** @hkvge

- **Краткое описание:**
  AWS Spot Instances — виртуальные машины с динамической ценой. При превышении указанного порога (`max_price`) инстанс немедленно прекращает работу.
  Проект предсказывает цену `g5.2xlarge` (GPU, популярный для QLoRA fine-tuning LLM) на **24 часа вперёд** с помощью LSTM.
  Сервис принимает текущую цену и `max_price`, возвращает прогноз и предупреждение — безопасно ли запускать джобу.

---

## 2. Структура проекта

```
project/
├── data/                   # Данные (parquet)
├── notebooks/              # Jupyter-ноутбуки: EDA, признаки, эксперименты
│   ├── 01_dataset_creating.ipynb
│   ├── 02_timeseries_eda.ipynb
│   ├── 03_features_and_baseline.ipynb
│   ├── 04_models_comparison.ipynb
│   └── 05_lstm_tuning.ipynb
├── src/                    # Исходный код
│   ├── features.py         # Feature engineering
│   ├── model.py            # LSTM + Dataset классы
│   ├── train.py            # Скрипт обучения
│   ├── service.py          # FastAPI сервис
│   └── Dockerfile
├── tests/                  # Тесты (pytest)
│   ├── test_features.py
│   ├── test_model.py
│   └── test_service.py
├── configs/
│   ├── training.yaml       # Гиперпараметры модели (Optuna best params)
│   └── .env.example        # Шаблон переменных окружения
├── artifacts/              # Артефакты обучения (lstm.pt, scaler.pkl, metrics.json)
├── requirements.txt
└── report.md
```

---

## 3. Требования и установка

- Python >= 3.10

```bash
cd project

# Создать виртуальное окружение
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Установить зависимости
pip install -r requirements.txt
```

---

## 4. Как запустить проект

### 4.1. Обучение модели

```bash
cd project
source .venv/bin/activate
python -m src.train
```

Вывод:
```
Device: mps
Training LSTM: input_size=22, hidden=64, layers=1, seq_len=14, epochs=50
  Epoch 10/50  loss=0.37010
  ...
  Epoch 50/50  loss=0.09558
Test → MAE=0.0562  RMSE=0.0798  MAPE=4.04%
Artifacts saved to artifacts/
```

Артефакты сохраняются в `artifacts/`:
- `lstm.pt` — веса модели
- `scaler.pkl` — нормализатор
- `metrics.json` — метрики на тестовой выборке

### 4.2. Запуск сервиса

```bash
cd project
source .venv/bin/activate
uvicorn src.service:app --port 8000
```

Сервис поднимается на `http://localhost:8000`.

Документация API (Swagger UI): `http://localhost:8000/docs`

### 4.3. Docker

```bash
cd project
docker build -f src/Dockerfile -t spot-forecaster .
docker run -p 8000:8000 spot-forecaster
```

### 4.4. Ключевые endpoints

| Endpoint | Метод | Описание |
|----------|-------|----------|
| `/health` | GET | Liveness check |
| `/predict` | POST | Предсказание цены на 24 ч вперёд |
| `/docs` | GET | Swagger UI |

**Пример запроса:**
```bash
curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"current_price": 1.01, "max_price": 1.10}'
```

**Пример ответа:**
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

---

## 5. Данные

- **Источник:** [Zenodo AWS Spot Price History](https://zenodo.org/records/18821638) — открытый датасет
- **Инстанс:** `g5.2xlarge`, OS: `Linux/UNIX`
- **Агрегация:** `max(cost)` , шаг 6 ч, forward fill
- **Период:** 2024-07 — 2026-02 (2 432 точки)

Файл `data/g5_2xlarge_6h_after_2024-07_with_features.parquet` включён в репозиторий и готов к использованию.

---

## 6. Тесты

```bash
cd project
source .venv/bin/activate
pytest tests/ -v
```

41 тест, покрывают:
- `test_features.py` — правильность вычисления признаков (лаги, синусы, праздники)
- `test_model.py` — форма тензоров, обучаемость, save/load модели
- `test_service.py` — HTTP-эндпоинты, валидация, safe/unsafe логика (без реальной модели)

---

## 7. Демонстрация на защите

1. **Показать структуру проекта** — `src/`, `notebooks/`, `configs/`
2. **Обучить модель:** `python -m src.train` → показать метрики (MAE, RMSE, MAPE)
3. **Запустить сервис:** `uvicorn src.service:app --port 8000`
4. **Открыть Swagger UI** на `http://localhost:8000/docs`
5. **Два сценария через Swagger:**
   - `current_price=1.01, max_price=1.15` → `is_safe: true`
   - `current_price=1.01, max_price=1.03` → `is_safe: false`, предупреждение
6. **Показать сравнение моделей** в `notebooks/04_models_comparison.ipynb`

---

## 8. Ограничения и развитие

- Данные не обновляются в реальном времени (нужна интеграция с AWS Spot Price API)
- Модель обучена только на `g5.2xlarge`
- Нет мониторинга дрейфа и автоматического переобучения

Подробнее — в `report.md`.

---

## 9. Оценка проекта

Ориентировочная оценка: 9-10/10 по чеклисту `self-checklist.md`.
