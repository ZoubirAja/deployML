from settings import (FEATURE_TARGET)
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import cross_validate, StratifiedKFold
import xgboost
import joblib
from config import calibrator, df_encoded
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier


#------------------------------------ Modele de prédiction-------------------------
def predict_model(X, y):
        # ---------------- SPLIT ----------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    # Adapter l'encoding sur poste pour reprendre un taux plutot que le nombre d'employée
    target_encoding = X_train.join(y_train).groupby('poste')['a_quitte_l_entreprise'].mean().to_dict()
    
    target_encoding = X_train.join(y_train).groupby('poste')['a_quitte_l_entreprise'].mean().to_dict()

    # Appliquer sur train ET test séparemment
    X_train['poste'] = X_train['poste'].map(target_encoding)
    X_test['poste'] = X_test['poste'].map(target_encoding)

    cols_to_scale = ['age', 'revenu_mensuel',
                     'distance_domicile_travail', 'augmentation_salaire_precedente_pourcentage']
    other_cols = X.columns.difference(cols_to_scale)
    
    transformers = []
    transformers.append(('scale', StandardScaler(), cols_to_scale))        
    transformers.append(('other', 'passthrough', other_cols))
    preprocessor = ColumnTransformer(transformers)
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
    
    # Récupérer les noms de colonnes dans le bon ordre
    feature_names = cols_to_scale + list(other_cols)
    
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

X = df_encoded.drop(columns=['a_quitte_l_entreprise', 'id_employee'])
y = df_encoded['a_quitte_l_entreprise']

predict_model(X, y)