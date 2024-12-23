from fastapi import FastAPI, File, UploadFile
from typing import Dict, List
from src.models_manager import ModelsManager
from src.models import TrainItem, PredictItem
import os
from minio import Minio
import mlflow

# Инициализация MinIO клиента с параметрами из переменных окружения
minio_access_key = os.getenv("MINIO_ROOT_USER", "minioadmin")
minio_secret_key = os.getenv("MINIO_ROOT_PASSWORD", "minioadmin")
minio_url = os.getenv("MINIO_URL", "minio:9000")

minio_client: Minio = Minio(
    minio_url,
    access_key=minio_access_key,
    secret_key=minio_secret_key,
    secure=False,
)

# Инициализация FastAPI приложения и ModelsManager
app: FastAPI = FastAPI()
models_manager: ModelsManager = ModelsManager()

# Настройка URI для отслеживания mlflow
mlflow.set_tracking_uri("http://mlflow:6000")
mlflow.set_experiment("FastAPI_Experiment")


@app.get("/get_available_methods")
async def get_methods() -> Dict[str, List[str]]:
    """
    Возвращает список доступных методов обучения из ModelsManager.

    Returns:
        Словарь с сообщением, содержащим доступные методы обучения.
    """
    return {"message": models_manager.get_available_classes()}


@app.get("/train")
async def train(train_item: TrainItem) -> Dict[str, str]:
    """
    Тренирует модель с использованием предоставленных данных и параметров.

    Args:
        train_item: Объект TrainItem, содержащий название модели, параметры, обучающие данные и метки.
    Returns:
        Словарь с сообщением об успешной тренировке и ID выполненного запуска в mlflow.
    """
    model_name: str = train_item.modelName
    parameters: Dict[str, float] = train_item.parameters
    X = train_item.train_data
    y = train_item.target_data

    with mlflow.start_run() as run:
        mlflow.log_params(parameters)
        model = models_manager.train(model_name, parameters, X, y)
        mlflow.sklearn.log_model(model, "model")
        mlflow.log_metric("accuracy", model.score(X, y))
        return {
            "message": f"Model {model_name} trained successfully!",
            "run_id": run.info.run_id,
        }


@app.get("/trained_models")
async def get_trained_models() -> Dict[str, List[str]]:
    """
    Возвращает список обученных моделей.

    Returns:
        Словарь с именами обученных моделей.
    """
    return {"trained_models": list(models_manager.trained_models.keys())}


@app.post("/predict")
async def predict(predict_item: PredictItem) -> Dict[str, List[float]]:
    """
    Выполняет предсказание на основе предоставленных данных и обученной модели.

    Args:
        predict_item: Объект PredictItem, содержащий название модели и тестовые данные.
    Returns:
        Словарь с предсказаниями.
    """
    model_name: str = predict_item.modelName
    X = predict_item.test_data
    y_pred: List[float] = models_manager.predict(model_name, X)
    return {"predictions": y_pred}


@app.get("/check_status")
async def check_status() -> Dict[str, str]:
    """
    Проверяет статус здоровья ModelsManager.

    Returns:
        Словарь с информацией о состоянии системы.
    """
    return models_manager.check_health()


@app.post("/upload_dataset")
def upload(file: UploadFile = File(...)) -> Dict[str, str]:
    """
    Загружает датасет в MinIO.

    Args:
        Загружаемый файл в формате UploadFile.
    Returns:
        Словарь с сообщением об успешной загрузке.
    """
    contents: bytes = file.file.read()
    file_name: str = file.filename
    with open(file_name, "wb") as f:
        f.write(contents)
    minio_client.fput_object("datasets", file_name, file_name)

    os.remove(file.filename)

    return {"message": f"Successfully uploaded {file_name}"}
