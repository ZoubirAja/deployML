import pytest

# ======== /predict_nouveau (données brutes) ========

def test_predict_valide(client, employe_valide):
    """Un employé valide doit retourner 200 avec une prédiction"""
    response = client.post("/predict_nouveau", json=employe_valide)

    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "probabilite_de_depart" in data
    assert "resultat" in data
    assert data["prediction"] in [0, 1]  # doit être 0 ou 1

def test_predict_champ_manquant(client):
    """Un champ obligatoire manquant doit retourner 422"""
    response = client.post("/predict_nouveau", json={"age": 28})  # plein de champs manquants
    assert response.status_code == 422

def test_predict_poste_invalide(client, employe_valide):
    """Un poste inconnu doit retourner une erreur de validation"""
    employe_valide["poste"] = "astronaute"  # poste qui n'existe pas
    response = client.post("/predict_nouveau", json=employe_valide)
    assert response.status_code == 422

def test_predict_frequence_invalide(client, employe_valide):
    """Une fréquence invalide doit retourner 422"""
    employe_valide["frequence_deplacement"] = "parfois"
    response = client.post("/predict_nouveau", json=employe_valide)
    assert response.status_code == 422

def test_predict_age_hors_limites(client, employe_valide):
    """Un age impossible doit retourner 422"""
    employe_valide["age"] = 150
    response = client.post("/predict_nouveau", json=employe_valide)
    assert response.status_code == 422

# ======== /predict/{id} (depuis BDD) ========

def test_predict_by_id_valide(client):
    """Un id existant doit retourner une prédiction"""
    response = client.post("/predict/1")
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data

def test_predict_by_id_inexistant(client):
    """Un id qui n'existe pas doit retourner une erreur claire"""
    response = client.post("/predict/999999")
    assert response.status_code == 200  # ton endpoint retourne 200 avec un message d'erreur
    data = response.json()
    assert "Erreur" in data
