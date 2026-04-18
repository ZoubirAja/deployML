from fastapi import FastAPI
from pydantic import BaseModel
from model import predict_model
from config import X, y
from employee import get_employee, EmployeeInput, prepare_dataframe
import joblib
import pandas as pd
#from init_db import init_db

# à appeler une seule fois pour créer et remplir la table
#init_db()

app = FastAPI()

@app.get("/score")
def getScoreModel():
    score = round(predict_model(X, y), 3)
    return {"Score de prédiction" : score}

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
    
    pipeline, calibrator, target_encoding, feature_names = joblib.load('model.pkl')

    # Appliquer le target encoding sur le poste
    employee_df['poste'] = employee_df['poste'].map(target_encoding)
    # Si le poste est inconnu du modèle → valeur par défaut
    if employee_df['poste'].isna().any():
        employee_df['poste'] = employee_df['poste'].fillna(0.5)

    prediction = pipeline.predict(employee_df)

    employee_transformed = pd.DataFrame(
        pipeline.named_steps['preprocessing'].transform(employee_df),
        columns=feature_names,
        index=employee_df.index
    )

    proba = calibrator.predict_proba(employee_transformed)
    
    resultat = "Va quitter l'entreprise" if prediction[0] == 1 else "Va rester dans l'entreprise"
    
    return {
        "id_employee": id_employee,
        "prediction": int(prediction[0]),
        "probabilite_de_depart": ("{0}%".format(round(proba[0][1]*100))),
        "resultat": resultat
    }

@app.post("/predict")
def predictEmployee(donnees: EmployeeInput):
    employee_df = prepare_dataframe(donnees)
    print(employee_df['non_satisfait'])

    pipeline, calibrator, target_encoding, feature_names = joblib.load('model.pkl')

    prediction = pipeline.predict(employee_df)

    employee_transformed = pd.DataFrame(
        pipeline.named_steps['preprocessing'].transform(employee_df),
        columns=feature_names,
        index=employee_df.index
    )

    proba = calibrator.predict_proba(employee_transformed)
    
    resultat = "Va quitter l'entreprise" if prediction[0] == 1 else "Va rester dans l'entreprise"
    
    return {
        "prediction": int(prediction[0]),
        "probabilite_de_depart": ("{0}%".format(round(proba[0][1]*100))),
        "resultat": resultat
    }

# -------- Profils de test --------

# Profil 1 : Risque élevé de départ
employe_risque_fort = {
    "heure_supplementaires": 1,
    "age": 28,
    "genre": 1,
    "revenu_mensuel": 2500,
    "poste": "ressources humaines",
    "nombre_experiences_precedentes": 5,
    "annee_experience_totale": 6,
    "annees_dans_l_entreprise": 1,
    "annees_dans_le_poste_actuel": 1,
    "nombre_participation_pee": 0,
    "nb_formations_suivies": 1,
    "distance_domicile_travail": 28,
    "niveau_education": 2,
    "frequence_deplacement": "frequent",
    "annees_depuis_la_derniere_promotion": 3,
    "annes_sous_responsable_actuel": 1,
    "departement": "rh",
    "augmentation_salaire_precedente_pourcentage": 11,
    "satisfaction_employee_environnement": 1,
    "satisfaction_employee_nature_travail": 1,
    "satisfaction_employee_equipe": 1,
    "satisfaction_employee_equilibre_pro_perso": 1,
    "note_evaluation_precedente": 4,
    "note_evaluation_actuelle": 4,
    "niveau_hierarchique_poste": 1,
    "statut_marital": 0,
    "domaine_etude_0": 0,
    "domaine_etude_1": 1,
    "domaine_etude_2": 0
}

# Profil 2 : Risque faible de départ
employe_stable = {
    "heure_supplementaires": 1,
    "age": 25,
    "genre": 0,
    "revenu_mensuel": 3000,
    "poste": "cadre commercial",
    "nombre_experiences_precedentes": 6,
    "annee_experience_totale": 20,
    "annees_dans_l_entreprise": 12,
    "annees_dans_le_poste_actuel": 6,
    "nombre_participation_pee": 5,
    "nb_formations_suivies": 4,
    "distance_domicile_travail": 5,
    "niveau_education": 4,
    "frequence_deplacement": "jamais",
    "annees_depuis_la_derniere_promotion": 1,
    "annes_sous_responsable_actuel": 8,
    "departement": "commercial",
    "augmentation_salaire_precedente_pourcentage": 20,
    "satisfaction_employee_environnement": 4,
    "satisfaction_employee_nature_travail": 4,
    "satisfaction_employee_equipe": 4,
    "satisfaction_employee_equilibre_pro_perso": 4,
    "note_evaluation_precedente": 3,
    "note_evaluation_actuelle": 3,
    "niveau_hierarchique_poste": 4,
    "statut_marital": 1,
    "domaine_etude_0": 1,
    "domaine_etude_1": 0,
    "domaine_etude_2": 0
}

# Profil 3 : Cas limite — champs optionnels absents
employe_minimal = {
    "heure_supplementaires": 1,
    "age": 35,
    "genre": 1,
    "revenu_mensuel": 4500,
    "poste": "manager",
    "nombre_experiences_precedentes": 3,
    "annee_experience_totale": 10,
    "annees_dans_l_entreprise": 4,
    "annees_dans_le_poste_actuel": 2,
    "nombre_participation_pee": 2,
    "nb_formations_suivies": 2,
    "distance_domicile_travail": 15,
    "niveau_education": 3,
    "frequence_deplacement": "occasionnel",
    "annees_depuis_la_derniere_promotion": 2,
    "annes_sous_responsable_actuel": 3,
    "departement": "consulting",
    "augmentation_salaire_precedente_pourcentage": 15
    # Pas de champs optionnels → valeurs médianes appliquées
}

# -------- Appels API --------
profils = {
    "Risque fort": employe_risque_fort,
    "Stable": employe_stable,
    "Minimal": employe_minimal
}

for nom, profil in profils.items():
    response = predictEmployee(EmployeeInput(**profil))
    print(f"\n--- {nom} ---")
    print(response)