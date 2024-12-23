from pydantic import BaseModel
from typing import Any, Dict, List


class TrainItem(BaseModel):
    """
    Модель данных для обучения модели машинного обучения.

    Атрибуты:
        modelName (str): Название модели (например, "RandomForest" или "LogisticRegression").
        parameters (dict): Параметры для настройки модели (например, {"n_estimators": 100}).
        train_data (list): Данные для обучения (например, список списков или массив).
        target_data (list): Целевые значения для обучения.
    """

    modelName: str
    parameters: Dict[str, Any]
    train_data: List
    target_data: List


class PredictItem(BaseModel):
    """
    Модель данных для предсказания с использованием обученной модели.

    Атрибуты:
        modelName (str): Название обученной модели, которая будет использоваться для предсказания.
        test_data (list): Данные для предсказания (например, список списков или массив).
    """

    modelName: str
    test_data: List
