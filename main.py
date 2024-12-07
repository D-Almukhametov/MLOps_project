from fastapi import FastAPI, File, UploadFile, HTTPException
from models import ClassModel
from pydantic import BaseModel
import os
from minio import Minio
import mlflow


minio_client = Minio("127.0.0.1:9000",
                     access_key="9YzOEjyBsjkTu58xxHDL",
                     secret_key='DvubBFsaATZ6jtfPnIclEt1Sihls1Oqz7mskRkEJ',
                     secure=False)



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
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("Fastapi_experiment")
    with mlflow.start_run():
        mlflow.log_params(train_item.parameters.dict())
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


@app.post("/upload_dataset")
def upload(file: UploadFile = File(...)):
    try:
        contents = file.file.read()
        with open(file.filename, 'wb') as f:
            f.write(contents)
        minio_client.fput_object('datasets', file.filename, file.filename)

        os.remove(file.filename)
    except Exception:
        raise HTTPException(status_code=500, detail='Something went wrong')
    finally:
        file.file.close()

    return {"message": f"Successfully uploaded {file.filename}"}