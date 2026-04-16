from fastapi import FastAPI
from pydantic import BaseModel
from model import predict_model, predict_model_classification
from config import X, y
from employee import get_employee
import joblib
#from init_db import init_db

# à appeler une seule fois pour créer et remplir la table
#init_db()

app = FastAPI()

@app.get("/score")
def getScoreModel():
    score = round(predict_model(X, y), 3)
    return {"Score de prédiction" : score}

@app.get("/classification/score")
def getScoreModel():
    score = round(predict_model_classification(X, y), 3)
    return {"Score de prédiction du 2eme modele" : score}

@app.post("/predict/{id_employee}")
def predictEmployee(id_employee: int):
    employee_df = get_employee(id_employee)
    if employee_df is None:
        return {
            "Erreur": "Aucun Employé trouvé pour cet id"
        }
    # Remettre les noms de colonnes originaux attendus par le pipeline
    employee_df = employee_df.rename(columns={
        "departement_consulting": "departement_Consulting",
        "departement_ressources_humaines": "departement_Ressources Humaines"
    })

    employee_df = employee_df.drop(columns=["a_quitte_l_entreprise"], errors="ignore")
    
    pipeline, calibrator = joblib.load('model.pkl')

    prediction = pipeline.predict(employee_df)
    employee_transformed = pipeline.named_steps['preprocessing'].transform(employee_df)
    proba = calibrator.predict_proba(employee_transformed)[:, 1][0]
    print(proba)
    
    resultat = "Va quitter l'entreprise" if prediction[0] == 1 else "Va rester dans l'entreprise"
    
    return {
        "id_employee": id_employee,
        "prediction": int(prediction[0]),
        "probabilite_de_depart": round(float(proba), 3),
        "resultat": resultat
    }