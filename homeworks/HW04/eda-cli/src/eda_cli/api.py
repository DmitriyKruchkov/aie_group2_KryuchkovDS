"""
HTTP-сервис для анализа качества датасетов на базе FastAPI.
"""
from __future__ import annotations

import time
from io import BytesIO
from typing import Any, Dict

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from .core import (
    compute_quality_flags,
    missing_table,
    summarize_dataset,
)

app = FastAPI(
    title="EDA Quality Service",
    description="HTTP-сервис для анализа качества датасетов",
    version="0.1.0",
)


class QualityRequest(BaseModel):
    """Запрос для оценки качества данных."""

    n_rows: int
    n_cols: int
    max_missing_share: float
    too_few_rows: bool = False
    too_many_columns: bool = False


class QualityResponse(BaseModel):
    """Ответ с оценкой качества данных."""

    ok_for_model: bool
    quality_score: float
    latency_ms: float
    flags: Dict[str, Any]


class QualityFlagsResponse(BaseModel):
    """Ответ с полным набором флагов качества."""

    flags: Dict[str, Any]


@app.get("/health")
async def health() -> Dict[str, str]:
    """
    Проверка работоспособности сервиса.
    """
    return {"status": "ok"}


@app.post("/quality", response_model=QualityResponse)
async def quality(request: QualityRequest) -> QualityResponse:
    """
    Оценка качества данных на основе входных параметров.

    Принимает JSON с параметрами датасета и возвращает оценку качества.
    """
    start_time = time.time()

    # Простая эвристика: ok_for_model если нет критических проблем
    ok_for_model = (
        not request.too_few_rows
        and not request.too_many_columns
        and request.max_missing_share < 0.5
    )

    # Простой скор качества
    quality_score = 1.0
    if request.too_few_rows:
        quality_score -= 0.2
    if request.too_many_columns:
        quality_score -= 0.1
    quality_score -= request.max_missing_share
    quality_score = max(0.0, min(1.0, quality_score))

    flags = {
        "too_few_rows": request.too_few_rows,
        "too_many_columns": request.too_many_columns,
        "too_many_missing": request.max_missing_share > 0.5,
        "max_missing_share": request.max_missing_share,
    }

    latency_ms = (time.time() - start_time) * 1000

    return QualityResponse(
        ok_for_model=ok_for_model,
        quality_score=quality_score,
        latency_ms=latency_ms,
        flags=flags,
    )


@app.post("/quality-from-csv", response_model=QualityResponse)
async def quality_from_csv(
    file: UploadFile = File(..., description="CSV-файл для анализа"),
) -> QualityResponse:
    """
    Оценка качества данных из загруженного CSV-файла.

    Читает CSV-файл, анализирует его и возвращает оценку качества.
    """
    start_time = time.time()

    try:
        # Читаем файл
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Файл пуст")

        # Парсим CSV
        try:
            df = pd.read_csv(BytesIO(contents))
        except Exception as e:
            raise HTTPException(
                status_code=400, detail=f"Не удалось прочитать CSV: {str(e)}"
            ) from e

        if df.empty:
            raise HTTPException(status_code=400, detail="CSV-файл не содержит данных")

        # Анализируем датасет
        summary = summarize_dataset(df)
        missing_df = missing_table(df)
        flags = compute_quality_flags(summary, missing_df, df)

        # Определяем ok_for_model
        ok_for_model = (
            not flags.get("too_few_rows", False)
            and not flags.get("too_many_columns", False)
            and not flags.get("too_many_missing", False)
        )

        quality_score = flags.get("quality_score", 0.0)

        latency_ms = (time.time() - start_time) * 1000

        # Формируем ответ с основными флагами
        response_flags = {
            "too_few_rows": flags.get("too_few_rows", False),
            "too_many_columns": flags.get("too_many_columns", False),
            "too_many_missing": flags.get("too_many_missing", False),
            "max_missing_share": flags.get("max_missing_share", 0.0),
        }

        return QualityResponse(
            ok_for_model=ok_for_model,
            quality_score=quality_score,
            latency_ms=latency_ms,
            flags=response_flags,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Внутренняя ошибка сервера: {str(e)}"
        ) from e


@app.post("/quality-flags-from-csv", response_model=QualityFlagsResponse)
async def quality_flags_from_csv(
    file: UploadFile = File(..., description="CSV-файл для анализа"),
) -> QualityFlagsResponse:
    """
    Полный набор флагов качества данных из загруженного CSV-файла.

    Возвращает все флаги качества, включая новые эвристики из HW03:
    - has_constant_columns
    - has_high_cardinality_categoricals
    - has_suspicious_id_duplicates
    - has_many_zero_values
    """
    try:
        # Читаем файл
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Файл пуст")

        # Парсим CSV
        try:
            df = pd.read_csv(BytesIO(contents))
        except Exception as e:
            raise HTTPException(
                status_code=400, detail=f"Не удалось прочитать CSV: {str(e)}"
            ) from e

        if df.empty:
            raise HTTPException(status_code=400, detail="CSV-файл не содержит данных")

        # Анализируем датасет
        summary = summarize_dataset(df)
        missing_df = missing_table(df)
        all_flags = compute_quality_flags(summary, missing_df, df)

        # Формируем ответ только с булевыми флагами (без детальных списков)
        response_flags = {
            "too_few_rows": all_flags.get("too_few_rows", False),
            "too_many_columns": all_flags.get("too_many_columns", False),
            "too_many_missing": all_flags.get("too_many_missing", False),
            "has_constant_columns": all_flags.get("has_constant_columns", False),
            "has_high_cardinality_categoricals": all_flags.get(
                "has_high_cardinality_categoricals", False
            ),
            "has_suspicious_id_duplicates": all_flags.get(
                "has_suspicious_id_duplicates", False
            ),
            "has_many_zero_values": all_flags.get("has_many_zero_values", False),
        }

        return QualityFlagsResponse(flags=response_flags)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Внутренняя ошибка сервера: {str(e)}"
        ) from e

