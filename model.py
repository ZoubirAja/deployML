from settings import (FEATURE_TARGET)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import category_encoders as ce

from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
                            make_scorer,
                            f1_score,
                            accuracy_score,
                            precision_score,
                            recall_score,
                            classification_report, 
                            confusion_matrix,
                            roc_auc_score,
                            precision_recall_curve, 
                            auc,
                            roc_curve)
from sklearn.tree import DecisionTreeClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.inspection import permutation_importance
import shap
from sklearn.model_selection import GridSearchCV
import xgboost


#------------------------------------ Modele de prédiction-------------------------
y_p = 0 
gt = 0
y_cus = 0

def predict_model(X, y, model=LogisticRegression(max_iter=10000, class_weight='balanced'),
                  scale_val=True,
                  cross_val_type=StratifiedKFold(),
                  scoring_metrics = {
                        "f1": make_scorer(f1_score, zero_division=0),
                        "precision": make_scorer(precision_score, zero_division=0),
                        "recall": make_scorer(recall_score, zero_division=0),
                        "roc_auc": make_scorer(roc_auc_score),
                        "accuracy": make_scorer(accuracy_score)
                    },
                  verbose=False,
                  precision_recall=False,
                  graph_roc_auc=False,
                  test_depth=False,
                  permutation=False,
                  show_waterfall=-1,
                  show_shap_scatter='None',
                  show_beeswarm=False,
                  show_grid=False,
                  show_counterfactuals = 0):

    global y_p, gt, y_cus
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
    if scale_val:
        transformers.append(('scale', StandardScaler(), cols_to_scale))
    else:
        transformers.append(('no_scale', 'passthrough', cols_to_scale))
        
    transformers.append(('other', 'passthrough', other_cols))
    preprocessor = ColumnTransformer(transformers)
    pipeline = Pipeline([
        ('preprocessing', preprocessor),
        ('model', model)
    ])

    # ---------------- PIPELINE ----------------
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_p = y_pred
    gt = y_test
    y_pred_train = pipeline.predict(X_train)
    
    if hasattr(model, "predict_proba"):
        y_scores = pipeline.predict_proba(X_test)[:, 1]
    else:
        y_scores = pipeline.decision_function(X_test)

    # --------------- Définition du seuil de prédiction -----------------
    precision_test, recall_test, thresholds_test = precision_recall_curve(
        y_test, y_scores
    )

    # On priorise le recall, on place donc le score minimun à 0.85
    idx = np.where(recall_test >= 0.5)[0][-1]
    best_threshold = thresholds_test[idx]

    y_pred_custom = (y_scores >= best_threshold).astype(int)
    y_cus = y_pred_custom

    

    if verbose:
        return f1_score(y_test, y_pred, zero_division=0)
        print("Accuracy :", accuracy_score(y_test, y_pred))
        print("F1 Score :", f1_score(y_test, y_pred, zero_division=0))
        print(classification_report(y_test, y_pred, zero_division=0))
        print("Matrice de confusion : ")
        print(confusion_matrix(y_test, y_pred))
        scores = cross_validate(
            pipeline,
            X,
            y,
            cv=cross_val_type,
            return_train_score=True,
            scoring=scoring_metrics,
        )

        for metric in scoring_metrics:
            print(
                "Average Train {metric} : {metric_value}".format(
                    metric=metric,
                    metric_value=np.mean(scores["train_" + metric]),
                )
            )
            print(
                "Train {metric} Standard Deviation : {metric_value}".format(
                    metric=metric, metric_value=np.std(scores["train_" + metric])
                )
            )
    
            print(
                "Average Test {metric} : {metric_value}".format(
                    metric=metric, metric_value=np.mean(scores["test_" + metric])
                )
            )
            print(
                "Test {metric} Standard Deviation : {metric_value}".format(
                    metric=metric, metric_value=np.std(scores["test_" + metric])
                )
            )

    if graph_roc_auc:
        fpr, tpr, thresholds = roc_curve(y_test, y_scores)
        plt.plot(fpr, tpr)
        plt.show()
        print(roc_auc_score(y_test, y_scores))

    # ---------------- Test du modèle avec plusieurs profondeurs pour un RandomForest ----------------
    if test_depth:    
        scores = []
        depth = np.arange(2, 30, 2)
        for d in depth:
            pipeline = Pipeline([
                ('preprocessing', preprocessor),
                ('model', model.__class__(max_depth=d))
            ])
        
            pipeline.fit(X_train, y_train)
        
            train_auc = roc_auc_score(y_train, pipeline.predict_proba(X_train)[:, 1])
            test_auc = roc_auc_score(y_test, pipeline.predict_proba(X_test)[:, 1])

            scores.append({
                'max_depth': d,
                'train': train_auc,
                'test': test_auc,
            })
        scores = pd.DataFrame(scores)
        
        plt.plot(scores['max_depth'], scores['train'], label='train')
        plt.plot(scores['max_depth'], scores['test'], label='test')
        plt.legend(loc='upper right')

    # ---------------- Graphe Precision Recall pour trouver le meilleur compromis ----------------
    if precision_recall:
        print(f"Seuil pour recall >= 0.85 : {best_threshold:.2f}")
        print("Matrice de confusion : ")
        print(confusion_matrix(y_test, y_pred_custom))

        baseline = y_test.mean() 
        plt.plot(recall_test, precision_test, label="Precision-Recall Curve")
        plt.axhline(y=baseline, color='red', linestyle='--', label=f"Baseline ({baseline:.0%})")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve Test Set")
        plt.legend()
        plt.show()
        
        auc_test = auc(recall_test, precision_test)
        print("Score AUC pour Test : ", auc_test)

        # ---------------- PERMUTATION POUR DETERMINER LA FEATURE IMPORTANCE ----------------
    if permutation:
        importances = pd.Series(
            pipeline.named_steps["model"].coef_.ravel(),
            index=X.columns
        ).sort_values(key=abs, ascending=False)
        
        r = permutation_importance(pipeline, X_train[X.columns], y_train,
                                   scoring="f1")
        
        permutation_importances = r.importances_mean
        perm_series = pd.Series(permutation_importances, index=X.columns)

        grouped_perm = {}
        for col, value in perm_series.items():
            if col.startswith("domaine_etude"):
                key = "domaine_etude"
            else:
                key = col
            grouped_perm[key] = grouped_perm.get(key, 0) + abs(value)
        
        grouped_perm = pd.Series(grouped_perm).sort_values()
        permutation_indices = np.argsort(grouped_perm.values)
        
        plt.figure(figsize=(10, 12))
        plt.title("Feature Importances")
        plt.barh(
            range(len(permutation_indices)),
            grouped_perm.values[permutation_indices],
            align='center'
        )
        plt.yticks(range(len(permutation_indices)), grouped_perm.index[permutation_indices])
        plt.xlabel("Permutation Importances")
        print("Les 10 features les plus influentes dans notre modèle : ")
        print((grouped_perm.tail(10)).sort_values(ascending=False))

    # ---------------- METRIQUE DE SHAPLEY WATERFALL ----------------
    
    if show_waterfall != -1:
        explanation = prepare_shapley(pipeline, X_test)
        shap.plots.waterfall(explanation[show_waterfall])

    # show_shap_scatter répresente la x feature la plus important ou le nom de la feature que l'on souhaite étudier
    if show_shap_scatter != 'None':
        explanation = prepare_shapley(pipeline, X_test)
        
        if isinstance(show_shap_scatter, int):
            feature_scatter = explanation.abs.mean(0).argsort[-(show_shap_scatter)]
            feature_name = explanation.feature_names[feature_scatter.values]
        else:
            feature_scatter = list(explanation.feature_names).index('other__' + show_shap_scatter)
            feature_name = show_shap_scatter
    
        shap.plots.scatter(explanation[:, feature_scatter], color=explanation, show=False)
        plt.title(f"{feature_name} dependence plot", color="g")
        plt.ylabel(f"SHAP value for the {feature_name} feature")
        
        plt.show()
        
    if show_beeswarm:
        explanation = prepare_shapley(pipeline, X_train)
        shap.plots.beeswarm(explanation, max_display=len(explanation.feature_names))

    # ---------------- GRIDSEARCHCV ----------------
    if show_grid:
        params = [
            {
                'model__loss': ['squared_hinge'],
                'model__C': [0.001, 0.01, 0.1, 1, 10, 100],
                'model__max_iter': [1000, 5000, 10000],
                'model__tol': [1e-4, 1e-3, 1e-2],
            },
        ]
        
        grid = GridSearchCV(
            pipeline,
            params,
            cv=cross_val_type,
            scoring='recall',
            verbose=1
        )

        grid.fit(X_train, y_train)

        print(f"Meilleurs params : {grid.best_params_}")
        print(f"Meilleur Recall score   : {grid.best_score_:.3f}")

    


def prepare_shapley(pipeline, X):
    # Récupérer les noms de features après preprocessing
    feature_names = pipeline.named_steps['preprocessing'].get_feature_names_out()
    
    # Transformer X_test
    X_transformed = pipeline.named_steps['preprocessing'].transform(X)
    
    # Créer un DataFrame avec les bons noms
    X_transformed_df = pd.DataFrame(X_transformed, columns=feature_names)
    
    tree_models = (
        xgboost.XGBClassifier, xgboost.XGBRegressor,
        RandomForestClassifier
        # ajoute tes modèles tree-based ici
    )
    linear_models = (
        LogisticRegression, LinearSVC
        # ajoute tes modèles linéaires ici
    )
    
    if isinstance(pipeline.named_steps['model'], tree_models):
        explainer = shap.TreeExplainer(pipeline.named_steps['model'])
    elif isinstance(pipeline.named_steps['model'], linear_models):
        explainer = shap.LinearExplainer(pipeline.named_steps['model'], X_transformed_df.astype(float))
    else:
        explainer = shap.KernelExplainer(pipeline.named_steps['model'].predict, shap.sample(X_transformed_df.astype(float), 100))
    
    return explainer(X_transformed_df.astype(float))
