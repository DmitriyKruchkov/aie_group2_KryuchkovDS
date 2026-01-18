# HW06 – Отчёт

> Файл: `homeworks/HW06/report.md`

## 1. Dataset
Использован `S06-hw-dataset-04.csv`. Размер: 25000 строк, 62 столбца (включая `id` и `target`). Целевая переменная `target`: класс 0 — 23770 (95.08%), класс 1 — 1230 (4.92%). Признаки: 60 числовых `f01`–`f60`, `id` исключён.

## 2. Protocol
Разделение train/test = 80/20, `random_state=42`, `stratify=y`.
Подбор параметров через `GridSearchCV` только на train, 5 фолдов, оптимизация по `roc_auc`.
Метрики: accuracy, F1 (binary), ROC-AUC; при дисбалансе ориентир на F1 и ROC-AUC.

## 3. Models
Базовые модели: DummyClassifier (`most_frequent`), LogisticRegression в Pipeline(StandardScaler).
Модели недели 6: DecisionTreeClassifier (подбор `max_depth`, `min_samples_leaf`), RandomForestClassifier (подбор `max_depth`, `max_features`, `min_samples_leaf`, `n_estimators=300`), GradientBoostingClassifier (подбор `n_estimators`, `learning_rate`, `max_depth`).

## 4. Results
Метрики на тесте: DummyClassifier — acc 0.951, F1 0.000, ROC-AUC 0.500;
LogisticRegression — acc 0.963, F1 0.429, ROC-AUC 0.834; DecisionTree — acc 0.966, F1 0.567, ROC-AUC 0.819;
RandomForest — acc 0.973, F1 0.634, ROC-AUC 0.897;
GradientBoosting — acc 0.975, F1 0.680, ROC-AUC 0.889.
Победитель по CV ROC-AUC — GradientBoostingClassifier (CV 0.888; тест: acc 0.975, F1 0.680, ROC-AUC 0.889).

## 5. Analysis
Устойчивость: оценка на train через CV, тест повторно не использовался (LogisticRegression — mean ROC-AUC 0.820, GradientBoosting — 0.888 +- (плюс-минус) 0.001 при 5 random_state).
Для лучшей модели матрица ошибок: TN=4744, FP=10, FN=114, TP=132;
при дисбалансе важнее контролировать FN, поэтому F1 и ROC-AUC информативнее accuracy.
Перестановочная важность (топ-15): f54, f25, f58, f13, f41, f11, f53, f33, f47, f38, f15, f43, f04, f52, f07.

## 6. Conclusion
Дерево решений требует контроля сложности, иначе происходит переобучение. Ансамбли дают более высокое качество на дисбалансных данных.
При дисбалансе ориентир — F1 и ROC-AUC, accuracy вторична.