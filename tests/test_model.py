import pytest
import joblib
import pandas as pd

def test_score_minimum():
    """Le F1 score sauvegardé doit être supérieur à 0.4"""
    _, _, _, _, score = joblib.load('model.pkl')  # si tu sauvegardes le score
    assert score >= 0.4, f"Score trop bas : {score}"

def test_pipeline_predit_binaire(model):
    """Le pipeline doit retourner 0 ou 1"""
    pipeline, _, target_encoding, feature_names, __ = model

    # Créer une ligne de données minimale
    row = pd.DataFrame([{col: 0 for col in feature_names}])
    prediction = pipeline.predict(row)

    assert prediction[0] in [0, 1]

def test_calibrateur_retourne_proba(model):
    """Le calibrateur doit retourner des probas entre 0 et 1"""
    pipeline, calibrator, target_encoding, feature_names, _ = model

    row = pd.DataFrame([{col: 0 for col in feature_names}])
    transformed = pd.DataFrame(
        pipeline.named_steps['preprocessing'].transform(row),
        columns=feature_names
    )
    proba = calibrator.predict_proba(transformed)

    assert proba[0].sum() == pytest.approx(1.0)  # les probas doivent sommer à 1
    assert all(0 <= p <= 1 for p in proba[0])