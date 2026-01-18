# Итог выполнения HW06 – ансамблевые модели на дисбалансных данных

## Кратко

В ноутбуке `HW06/HW06.ipynb` решается задача бинарной классификации для таргета
`target` на наборе признаков `f01–f60`. Сравниваются несколько моделей, ведётся
подбор гиперпараметров и формируется отчёт `report.md`.

## Структура

```text
homeworks/
  HW06/
    HW06.ipynb
    README.md
    S06-hw-dataset-04.csv
    report.md
    artifacts/
      best_model.joblib
      best_model_meta.json
      metrics_test.json
      search_summaries.json
      figures/
        roc_curve_best_model.png
        pr_curve_best_model.png
        confusion_matrix_best_model.png
        confusion_matrix_best_model_normalized.png
        permutation_importance_best_model.png
```

## Данные

Файл: `S06-hw-dataset-04.csv`

Сводка по датасету (из `report.md`):

* 25000 строк, 62 столбца (включая `id` и `target`)
* целевая переменная `target`: класс 0 — 23770 (95.08%), класс 1 — 1230 (4.92%)
* признаки: 60 числовых `f01–f60`, `id` исключён из признаков

## Как воспроизвести

### 1) Зависимости

Нужны библиотеки (минимально):

* `pandas`, `numpy`, `matplotlib`
* `scikit-learn`

Установка (пример через pip):

```bash
pip install pandas numpy matplotlib scikit-learn
```

Если используешь `uv`, то `pyproject.toml` лежит в корне репозитория,
поэтому зависимости ставятся так:

```bash
cd <корень-репозитория>
uv sync
```

### 2) Запуск

Открой ноутбук `homeworks/HW06/HW06.ipynb` и выполни все ячейки.

Пример через `uv` (при наличии Jupyter):

```bash
cd <корень-репозитория>
uv run jupyter lab
```

## Методология

* разбиение train/test = 80/20, `random_state=42`, `stratify=y`
* подбор параметров через `GridSearchCV`, оптимизация по `ROC-AUC`
* модели: DummyClassifier, LogisticRegression, DecisionTree, RandomForest, GradientBoosting
* метрики: accuracy, F1 (binary), ROC-AUC; акцент на F1 и ROC-AUC при дисбалансе

## Результаты на тесте

Из `report.md`:

| Модель                 | Accuracy | F1    | ROC-AUC |
| ---------------------- | -------- | ----- | ------- |
| DummyClassifier        | 0.951    | 0.000 | 0.500   |
| LogisticRegression     | 0.963    | 0.429 | 0.834   |
| DecisionTree           | 0.966    | 0.567 | 0.819   |
| RandomForest           | 0.973    | 0.634 | 0.897   |
| GradientBoosting       | 0.975    | 0.680 | 0.889   |

Лучший по CV ROC-AUC: `GradientBoostingClassifier` (CV 0.888).
Артефакты модели и графики сохранены в `homeworks/HW06/artifacts/`.

## Вывод

Ансамбли показывают наилучшее качество на дисбалансных данных.
При выборе модели ключевыми метриками являются F1 и ROC-AUC, а accuracy вторична.
