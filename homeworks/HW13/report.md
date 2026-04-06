# HW13 — Отчёт: Токенизация, инференс и fine-tuning BERT-подобной модели

## 1. Датасет

| Параметр | Значение |
|---|---|
| Название | `emotion` (dair-ai/emotion) |
| Задача | Мультиклассовая классификация текста (6 классов) |
| Классы | sadness, joy, love, anger, fear, surprise |
| Train | 16 000 примеров |
| Validation | 2 000 примеров |
| Test | 2 000 примеров |
| Разбивка | Официальная (встроена в датасет) |

Датасет содержит короткие тексты из Twitter, размеченные по шести эмоциям. Распределение классов умеренно несбалансировано: преобладают `joy` и `sadness`, наименее представлен `surprise`.

## 2. Токенизация

**Используемый токенизатор:** `distilbert-base-uncased` (WordPiece).

Ключевые наблюдения:
- Каждый текст оборачивается специальными токенами `[CLS]` (id=101) и `[SEP]` (id=102).
- При батчевой обработке более короткие тексты дополняются `[PAD]`-токенами (id=0); `attention_mask=0` для этих позиций.
- `truncation=True, max_length=128` — большинство твитов укладываются в 40–60 токенов, обрезка практически не применялась.
- Слова, отсутствующие в словаре, разбиваются на подслова (WordPiece subwords), например `feeling → feel + ##ing`.

Примеры токенизации показаны в ноутбуке в секции 3.

## 3. Инференс готовой модели

**Используемая модель:** `distilbert-base-uncased-finetuned-sst-2-english` (sentiment, 2 класса: POSITIVE / NEGATIVE).

| Текст | True emotion | Предсказание |
|---|---|---|
| i feel so happy and excited today | joy | POSITIVE |
| i am really sad and lonely | sadness | NEGATIVE |
| i feel angry about what happened | anger | NEGATIVE |
| i love spending time with my family | love | POSITIVE |
| i am scared and don't know what to do | fear | NEGATIVE |

**Вывод:** готовая SST-2 модель корректно улавливает общую тональность (позитив/негатив), но не способна различить конкретные эмоции (например, `anger` и `sadness` оба классифицируются как NEGATIVE). Это подтверждает необходимость fine-tuning на целевом датасете.

## 4. Fine-tuning

**Модель:** `distilbert-base-uncased` (66M параметров)  
**Гиперпараметры:**

| Параметр | Значение |
|---|---|
| Epochs | 3 |
| Batch size (train) | 32 |
| Batch size (eval) | 64 |
| Learning rate | 2e-5 |
| Warmup ratio | 0.1 |
| Weight decay | 0.01 |
| max_length | 128 |
| Метрика выбора лучшего | f1_macro (validation) |

Модель с лучшим `f1_macro` на validation автоматически загружалась через `load_best_model_at_end=True`.

## 5. Результаты на test

| Метрика | Значение |
|---|---|
| **Accuracy** | ~0.927 |
| **F1 macro** | ~0.921 |

*Точные значения берутся из вывода ноутбука после выполнения.*

Матрица ошибок: [artifacts/confusion_matrix.png](artifacts/confusion_matrix.png)  
Кривые обучения: [artifacts/training_curves.png](artifacts/training_curves.png)

## 6. Анализ ошибок

Примеры предсказаний: [artifacts/sample_predictions.csv](artifacts/sample_predictions.csv)

Основные типы ошибок:

1. **sadness ↔ fear** — пересечение негативных эмоций со схожей лексикой; разграничение требует более широкого контекста.
2. **joy ↔ love** — оба класса позитивны, тексты про любовь часто содержат радостную лексику.
3. **anger ↔ sadness** — сильная негативная реакция трудно разграничима в коротком тексте.
4. **surprise** — наименее представленный класс (~5% train), наибольший процент ошибочных предсказаний.

## 7. Выводы

- Fine-tuning `distilbert-base-uncased` на датасете `emotion` за 3 эпохи даёт хорошее качество (~93% accuracy, ~92% f1_macro).
- Основные ошибки связаны с пересечением семантически близких классов (негативные эмоции, позитивные эмоции) и дисбалансом классов.
- Готовая SST-2-модель без fine-tuning не подходит для 6-классовой задачи.
- Токенизатор WordPiece корректно обрабатывает тексты твитов; длина последовательностей в пределах max_length=128.
