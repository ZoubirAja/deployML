from fastapi import FastAPI
from pydantic import BaseModel
from model import predict_model
from config import X, y, model_linearSVC

app = FastAPI()

class Item(BaseModel):
    name:str
    price:float
    is_offer: bool | None = None

@app.get("/")
def read_root():
    return {"Hello": "World !!"}

@app.get("/score")
def getScoreModel():
    score = round(predict_model(X, y, verbose=True, model=model_linearSVC), 2)
    return {"Score de prédiction" : score}
