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
from config import calibrator
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier


#------------------------------------ Modele de prédiction-------------------------
def predict_model(X, y):
        # ---------------- SPLIT ----------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.2, random_state=42, stratify=y
    )
    # Adapter l'encoding sur poste pour reprendre un taux plutot que le nombre d'employée
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

    X_test_transf = pipeline.named_steps['preprocessing'].transform(X_test)
    calibrator.fit(X_test_transf, y_test)
    # On sauvegarde le modele pour l'appeler via l'API
    joblib.dump((pipeline, calibrator), 'model.pkl')

    y_pred = pipeline.predict(X_test)
    #y_scores = pipeline.predict_proba(X_test)

    return f1_score(y_test, y_pred, zero_division=0)

#------------------------------------ Modele de prédiction-------------------------
def predict_model_classification(X, y):
    model = XGBClassifier(
        # Gestion du déséquilibre
        scale_pos_weight=5,          # ratio classe_0 / classe_1 = 247/47

        # Anti-overfitting (ton vrai problème avec XGBoost)
        max_depth=3,                 # arbres peu profonds
        n_estimators=200,            # assez d'arbres mais pas trop
        learning_rate=0.05,          # apprentissage lent = meilleure généralisation
        subsample=0.8,               # 80% des lignes par arbre
        colsample_bytree=0.8,        # 80% des features par arbre
        min_child_weight=5,          # noeud minimum 5 exemples (protège classe minoritaire)
        gamma=1,                     # regularisation sur les splits

        # Objectif
        objective='binary:logistic',
        eval_metric='aucpr',         # AUC Precision-Recall, mieux que AUC-ROC sur déséquilibre
        random_state=42
    )
        # ---------------- SPLIT ----------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.2, random_state=42, stratify=y
    )
    # Adapter l'encoding sur poste pour reprendre un taux plutot que le nombre d'employée
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
        ('model', model)
    ])

    # ---------------- PIPELINE ----------------
    pipeline.fit(X_train, y_train)
    # On sauvegarde le modele pour l'appeler via l'API
    joblib.dump(pipeline, 'model_classification.pkl')

    y_pred = pipeline.predict(X_test)
    y_scores = pipeline.predict_proba(X_test)[:, 1]

    return f1_score(y_test, y_pred, zero_division=0)