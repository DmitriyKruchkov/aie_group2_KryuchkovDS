# Итог выполнения HW07 – кластеризация и внутренние метрики

## Кратко

В ноутбуке `HW07/HW07.ipynb` выполнены эксперименты по кластеризации на трёх
синтетических датасетах (01, 02, 04). Сравниваются KMeans и DBSCAN с подбором
параметров, считаются внутренние метрики и строятся PCA-визуализации. Итоги
оформлены в `report.md`, артефакты сохранены в `artifacts/`.

## Структура

```text
homeworks/
  HW07/
    HW07.ipynb
    README.md
    report.md
    data/
      S07-hw-dataset-01.csv
      S07-hw-dataset-02.csv
      S07-hw-dataset-04.csv
    artifacts/
      metrics_summary.json
      best_configs.json
      stability_kmeans_ds1.json
      labels/
        labels_hw07_ds1.csv
        labels_hw07_ds2.csv
        labels_hw07_ds4.csv
      figures/
        ds1_kmeans_silhouette_vs_k.png
        ds2_kmeans_silhouette_vs_k.png
        ds4_kmeans_silhouette_vs_k.png
        ds1_pca_best.png
        ds2_pca_best.png
        ds4_pca_best.png
```

## Данные

Использованы 3 датасета:

* `S07-hw-dataset-01.csv` – числовые признаки в разных шкалах + шум
* `S07-hw-dataset-02.csv` – нелинейная структура + выбросы + шумовой признак
* `S07-hw-dataset-04.csv` – высокая размерность + категориальные признаки + пропуски

Во всех датасетах есть колонка `sample_id` (не используется как признак).

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

Открой ноутбук `homeworks/HW07/HW07.ipynb` и выполни все ячейки.

Пример через `uv` (при наличии Jupyter):

```bash
cd <корень-репозитория>
uv run jupyter lab
```

## Методология

* препроцессинг: `StandardScaler`, `SimpleImputer` (числовые),
  `OneHotEncoder` (категориальные в dataset-04)
* модели: KMeans (поиск `k`), DBSCAN (поиск `eps`, `min_samples`)
* метрики: silhouette / Davies-Bouldin / Calinski-Harabasz
* визуализация: PCA(2D) scatter для лучшего решения на каждом датасете
* устойчивость: 5 запусков KMeans для dataset-01, ARI между разбиениями

## Результаты

Лучшие конфигурации (см. `artifacts/best_configs.json`):

* dataset-01: KMeans `k=2`
* dataset-02: DBSCAN `eps=1.0`, `min_samples=20`
* dataset-04: DBSCAN `eps=2.5`, `min_samples=20`

Метрики и детали по каждому датасету находятся в `report.md` и
`artifacts/metrics_summary.json`.

## Вывод

HW07 показывает различия между KMeans и DBSCAN на данных с разной геометрией,
а также важность препроцессинга, корректного подбора параметров и метрик
качества в задачах кластеризации без истинных меток.
