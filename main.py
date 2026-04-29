from fastapi import FastAPI
from employee import get_employee, EmployeeInput, prepare_dataframe, get_employees_groupe, log_prediction
import joblib
import pandas as pd
from minio_client import download_model
from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import APIKeyHeader
import os

API_KEY_HEADER = APIKeyHeader(name="X-API-Key")

def verify_api_key(key: str = Depends(API_KEY_HEADER)):
    if key != os.getenv("API_KEY"):
        raise HTTPException(
            status_code=403,
            detail="API Key invalide"
        )

@asynccontextmanager
async def lifespan(app: FastAPI):
    download_model()   # ← télécharge depuis MinIO au démarrage de l'API
    yield

app = FastAPI(lifespan=lifespan)

pipeline, calibrator, target_encoding, feature_names, score = joblib.load('model.pkl')

@app.get("/")
def root():
    return {"message": "Hello World ! \n deployML API tourne !!"}

@app.post("/predict/{id_employee}", dependencies=[Depends(verify_api_key)])
def predict_by_id(id_employee: int):
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

    # Appliquer le target encoding sur le poste
    employee_df['poste'] = employee_df['poste'].map(target_encoding)
    # Si le poste est inconnu du modèle → valeur par défaut
    if employee_df['poste'].isna().any():
        employee_df['poste'] = employee_df['poste'].fillna(0.5)

    return run_prediction(employee_df, id_employee)

@app.post("/predict_nouveau", dependencies=[Depends(verify_api_key)])
def predict_from_data(donnees: EmployeeInput):
    employee_df = prepare_dataframe(donnees)

    return run_prediction(employee_df)

def run_prediction(employee_df, id_employee=None):
    prediction = pipeline.predict(employee_df)

    employee_transformed = pd.DataFrame(
        pipeline.named_steps['preprocessing'].transform(employee_df),
        columns=feature_names,
        index=employee_df.index
    )

    proba = calibrator.predict_proba(employee_transformed)
    
    resultat = "Va quitter l'entreprise" if prediction[0] == 1 else "Va rester dans l'entreprise"
    
    # Sauvegarde résultalt
    log_prediction(
        id_employee=id_employee,
        prediction=int(prediction[0]),
        probabilite=round(proba[0][1]*100),
        resultat=resultat
    )

    return {
        "prediction": int(prediction[0]),
        "probabilite_de_depart": ("{0}%".format(round(proba[0][1]*100))),
        "resultat": resultat
    }

@app.post("/predict/group/{poste}")
def predict_poste(poste: str):
    # Appliquer le target encoding sur le poste
    poste_encoded = target_encoding.get(poste.lower(), .5)

    employee_df = get_employees_groupe(poste_encoded)
    if employee_df is None:
        return {"Erreur": "Aucun employé trouvé pour ce poste"}
    
    ids = employee_df['id_employee'].tolist()
    
    employee_df = employee_df.rename(columns={
        "departement_consulting": "departement_Consulting",
        "departement_ressources_humaines": "departement_Ressources Humaines"
    })
    employee_df = employee_df.drop(columns=["a_quitte_l_entreprise"], errors="ignore")
    
    prediction = pipeline.predict(employee_df)
    # Probas pour chaque employé
    employee_transformed = pd.DataFrame(
        pipeline.named_steps['preprocessing'].transform(employee_df),
        columns=feature_names,
        index=employee_df.index
    )
    proba = calibrator.predict_proba(employee_transformed)
    idx_depart = list(calibrator.classes_).index(1)
    probas_depart = proba[:, idx_depart]  # tableau de probas pour tous les employés

    # Top 5 à risque
    top5_idx = probas_depart.argsort()[::-1][:5]
    top5 = [{
                "id_employee": ids[i],
                "probabilite_de_depart": f"{round(probas_depart[i] * 100)}%"
            }
            for i in top5_idx]

    return {
        "poste": poste,
        "nombre_employes": len(prediction),
        "nombre_departs_prevus": int(prediction.sum()),
        "taux_de_depart_prevu": f"{round(prediction.sum() / len(prediction) * 100)}%",
        "taux_de_risque_moyen": f"{round(probas_depart.mean() * 100)}%",
        "top5_employes_a_risque": top5
    }