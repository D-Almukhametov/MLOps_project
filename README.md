Структура проекта
Проект состоит из трех основных файлов:
- models/models_manager.py: Класс с обучением моделей машинного обучения
- models/models.py: Модели модуля pydantic
- app.py: FAST API


Запуск сервера FastAPI
```
uvicorn main:app --reload
```


Доступ к API:  
Сервер будет доступен по адресу: http://127.0.0.1:8000

Примеры параметров для обучения:

```
{
  "modelName": "RandomForest",
  "parameters": {},
  "train_data": [[0], [1], [2], [3]],
  "target_data": [0, 1, 1, 0]
}
```

```
{
    "modelName":"LogisticRegression",
    "parameters":{
        "C":2.0
    },
    "train_data":[[0], [1], [1], [1], [0], [1], [0], [1]],
    "target_data":[0, 1, 1, 1, 0, 1, 0, 1]
}
```
Примеры данных для предсказания
```
{
  "modelName": "RandomForest",
  "test_data": [[1], [2]]
  }
```

Командлы для Docker:
```
docker build -t my_fastapi:latest -f Dockerfile_fastapi .
docker-compose up -d 
```

Все эксперименты можно проводить здесь: 127.0.0.1:8000/docs
Выполнение предсказаний
Через интерфейс Streamlit выберите модель и задайте значения признаков.
Нажмите кнопку "Let's check".
Приложение отправит запрос на сервер и отобразит гистограмму предсказаний.
Эндпоинты API
GET /get_available_methods: Получить список доступных моделей для обучения.
POST /train: Обучить модель с предоставленными данными.
GET /trained_models: Получить список уже обученных моделей.
POST /predict: Получить предсказания от обученной модели.
GET /check_status: Проверить статус сервера и наличие обученных моделей.
Пример использования API

Обучение модели
curl -X POST "http://127.0.0.1:8000/train" -H "Content-Type: application/json" -d '{
  "modelName": "RandomForest",
  "parameters": {},
  "X": [[0], [1], [2], [3]],
  "y": [0, 1, 1, 0]
}'

Предсказание
curl -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" -d '{
  "modelName": "RandomForest",
  "X": [[1], [2]]
}'

Замечания:
Убедитесь, что директория models существует в корневой папке проекта для сохранения обученных моделей.
Данные для обучения (X, y) и предсказания (X) должны быть в формате списка.
Перед использованием приложения Streamlit надо убедиться, что сервер FastAPI запущен и модель обучена.