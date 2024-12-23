from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import sklearn
import pickle
import os
import logging
from typing import Dict, List, Any
import psutil

logging.basicConfig(level=logging.INFO)


class ModelsManager:
    """
    Класс для работы с моделями машинного обучения.
    """

    def __init__(self, models_dir="models/"):
        """
        Args:
            models_dir: директория, где хранятся обученные модели
        Attributes:
            trained_models: словарь с обученными моделями
            logger: логгер
        """
        self.models_dir = models_dir
        self.logger = logging.getLogger("ModelsManager")
        self.trained_models = self._load_models()

    def _load_models(self) -> Dict[str, sklearn.base.BaseEstimator]:
        """Загружает модели из файловой системы."""
        models = {}
        for file in os.listdir(self.models_dir):
            if file.endswith(".pickle"):
                model_name = file.split(".pickle")[0]
                with open(os.path.join(self.models_dir, file), "rb") as f:
                    model = pickle.load(f)
                    if isinstance(model, sklearn.base.BaseEstimator):
                        models[model_name] = model
                        self.logger.info(f"Модель '{model_name}' успешно загружена.")
        return models

    def get_available_classes(self) -> List[str]:
        """
        Возвращает список доступных для обучения моделей.

        Returns:
            List[str]: Список строк с именами доступных моделей.
        """
        return ["RandomForest", "LogisticRegression"]

    def train(
        self,
        model_name: str,
        parameters: Dict[str, Any],
        train_data: Any,
        target_data: Any,
    ) -> str:
        """
        Обучает модель на переданных данных и сохраняет её в файл.

        Args:
        model_name (str): Имя модели, которое будет использовано для её
                создания.
        parameters (Dict[str, Any]): Параметры для настройки модели.
        train_data (Any): Данные для обучения модели.
        target_data (Any): Целевые данные.

        Returns:
            str: Сообщение о результате обучения модели.
        """
        if model_name == "RandomForest":
            model = RandomForestClassifier(**parameters)
        elif model_name == "LogisticRegression":
            model = LogisticRegression(**parameters)
        else:
            raise ValueError(
                f"Модель '{model_name}' не поддерживается.\
                             Выберите из {self.get_available_classes()}."
            )

        # Обучение модели
        model.fit(train_data, target_data)
        self.trained_models[model_name] = model

        # Сохранение модели на диск
        path = "models/" + model_name + ".pickle"
        with open(path, mode="wb+") as f:
            pickle.dump(model, f)

        self.logger.info(f"Модель {model_name} была обучена")
        return model

    def predict(self, model_name: str, test_data: Any) -> List:
        """
        Использует обученную модель для предсказания на новых данных.

        Параметры:
        - model_name (str): Имя модели, которое будет
                использовано для предсказания.
        - test_data (Any): Данные, на которых нужно сделать предсказание.

        Возвращает:
            List: Список предсказанных значений.
        """
        # Проверяем, обучена ли модель
        if model_name not in self.trained_models:
            raise KeyError(
                f"Модель '{model_name}' не найдена в обученных моделях.\
                           Сначала обучите модель с помощью метода train."
            )

        model = self.trained_models[model_name]
        predictions = model.predict(test_data).tolist()
        return predictions

    def check_health(self) -> Dict[str, Any]:
        """
        Проверяет статус сервиса, включая использование оперативной памяти.

        Returns:
            Dict[str, Any]: Метрики состояния системы.
        """
        memory_info = psutil.virtual_memory()
        health_status = {
            "status": "Healthy",
            "memory_used, Gb": memory_info.used / (1024**3),
            "memory_total, Gb": memory_info.total / (1024**3),
            "loaded_models": list(self.trained_models.keys()),
        }
        self.logger.info("Проверка состояния сервиса выполнена.")
        return health_status
