# train.py — à lancer manuellement : python train.py
from config import X, y
from model import predict_model

score = predict_model(X, y)
print(f"Modèle entraîné — F1 score : {score}")