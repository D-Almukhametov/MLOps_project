from fastapi import FastAPI
from models import ClassModel
from pydantic import BaseModel


class TrainItem(BaseModel):
    modelName: str
    parameters: dict
    X: list
    y: list


class PredictItem(BaseModel):
    modelName: str
    X: list


app = FastAPI()
classModel = ClassModel()


@app.get("/get_available_methods")
async def get_methods():
    return {"message": classModel.get_available_classes()}


@app.get("/train")
async def train(train_item: TrainItem):
    model_name = train_item.modelName
    parameters = train_item.parameters
    X = train_item.X
    y = train_item.y
    return {"message": classModel.train(model_name, parameters, X, y)}


@app.get("/trained_models")
async def get_trained_models():
    return {"trained_models": list(classModel.trained_models.keys())}

@app.get("/predict")
async def predict(predict_item: PredictItem):
    model_name = predict_item.modelName
    X = predict_item.X
    y_pred = classModel.predict(model_name, X)
    return {"predictions": y_pred}


@app.get("/check_status")
async def check_status():
    if len(list(classModel.trained_models.keys())) == 0:
        return {"message": 'Initialized'}
    return {"message": "Working"}