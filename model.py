from settings import (FEATURE_TARGET)
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_validate, StratifiedKFold
import joblib
from config import calibrator, df_encoded
from sklearn.svm import LinearSVC


#------------------------------------ Modele de prédiction-------------------------
def predict_model(X, y):
        # ---------------- SPLIT ----------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    # ← Calculé sur TOUT le dataset pour matcher la BDD
    # Cela créer un léger data leakeage mais vu la taille de nos données ça reste acceptable
    target_encoding = X.join(y).groupby('poste')['a_quitte_l_entreprise'].mean().to_dict()
    target_encoding = {k.lower(): v for k, v in target_encoding.items()}

    X_train = X_train.copy()
    X_test = X_test.copy()
    
    # Appliquer sur train ET test séparemment
    X_train['poste'] = X_train['poste'].str.lower().map(target_encoding)
    X_test['poste'] = X_test['poste'].str.lower().map(target_encoding)

    cols_to_scale = ['age', 'revenu_mensuel',
                     'distance_domicile_travail', 'augmentation_salaire_precedente_pourcentage']
    other_cols = X_train.columns.difference(cols_to_scale)
    # Récupérer les noms de colonnes dans le bon ordre
    feature_names = cols_to_scale + list(other_cols)
    
    preprocessor = ColumnTransformer([
        ('scale', StandardScaler(), cols_to_scale),
        ('other', 'passthrough', other_cols)
    ])
    pipeline = Pipeline([
        ('preprocessing', preprocessor),
        ('model', LinearSVC(class_weight='balanced'))
    ])

    # ---------------- PIPELINE ----------------
    pipeline.fit(X_train, y_train)

        # Calibrer les données : Le linearSvc ne donne pas de proba
    # Avec un calibrateur on va écraser les données extremes puis mettre sur une échelle de 100
    # Pour que ce soit on doit faire ce calibrage sur les datas preprocessées 
    # (on utilise donc l'étape preprocessing du pipeline)
    
    # Reconvertir en DataFrame
    X_calib = pd.DataFrame(
        pipeline.named_steps['preprocessing'].transform(X_train),
        columns=feature_names,
        index=X_train.index
    )
    calibrator.fit(X_calib, y_train)
    # On sauvegarde le modele pour l'appeler via l'API
    joblib.dump((pipeline, calibrator, target_encoding, feature_names), 'model.pkl')

    y_pred = pipeline.predict(X_test)

    return f1_score(y_test, y_pred, zero_division=0)