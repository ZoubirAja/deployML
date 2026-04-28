import pytest
from fastapi.testclient import TestClient
from main import app
import joblib

# ---- Client de test FastAPI ----
@pytest.fixture
def client():
    # TestClient simule des vraies requêtes HTTP sans lancer le serveur
    with TestClient(app) as c:
        yield c

# ---- Profil employé valide réutilisable ----
@pytest.fixture
def employe_valide():
    return {
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
        "augmentation_salaire_precedente_pourcentage": 11
    }

# ---- Modèle chargé une seule fois ----
@pytest.fixture(scope="session")  # scope="session" = chargé une seule fois pour tous les tests
def model():
    pipeline, calibrator, target_encoding, feature_names, score = joblib.load('model.pkl')
    return pipeline, calibrator, target_encoding, feature_names, score