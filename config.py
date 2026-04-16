from settings import (FEATURE_TARGET, create_new_feature)
import pandas as pd
import numpy as np
import category_encoders as ce

from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from sklearn.calibration import CalibratedClassifierCV

#---------------- Exrtraction des données et fusion ------------------
sondage = pd.read_csv('datas/extrait_sondage.csv')
evaluation = pd.read_csv('datas/extrait_eval.csv')
sirh = pd.read_csv('datas/extrait_sirh.csv')

evaluation['id_employee'] = evaluation['eval_number'].str[2:].astype(int)
df = evaluation.merge(sirh)
df = df.merge(sondage, left_on='id_employee', right_on='code_sondage')
df = df.drop(columns=['nombre_heures_travailless', 'nombre_employee_sous_responsabilite', 'ayant_enfants'], errors='ignore')


#--------------- Encoding Feature ---------------------
df_encoded = df.copy()

# ------------- Encodage de la feature frequence_deplacement
deplacement_mapping = {
    'Aucun': 0,
    'Occasionnel': 1,
    'Frequent': 2
}
encoder = ce.OrdinalEncoder(mapping=[{'col': 'frequence_deplacement', 'mapping': deplacement_mapping}])
df_encoded = encoder.fit_transform(df)

# ------------- Encodage de la feature statut_marital
martial_mapping = {
    'Célibataire': 0,
    'Marié(e)': 1,
    'Divorcé(e)': 2
}
encoder = ce.OrdinalEncoder(mapping=[{'col': 'statut_marital', 'mapping': martial_mapping}])
df_encoded = encoder.fit_transform(df_encoded)

# ------------- Encodage de la feature departement
df_encoded = pd.get_dummies(df_encoded, columns=['departement'], drop_first=True)

# ------------- Encodage de la feature genre
df_encoded['genre'] = df_encoded['genre'].apply(lambda x: 0 if x == 'M' else 1)

# ------------- Encodage de la feature heure_supplementaires
df_encoded['heure_supplementaires'] = df_encoded['heure_supplementaires'].apply(lambda x: 0 if x == 'Non' else 1)

# ------------- Encodage de la feature augementation_salaire_precedente
df_encoded['augmentation_salaire_precedente_pourcentage'] = df_encoded['augementation_salaire_precedente'].str[:-1].astype(int)

# ------------- Encodage de la feature a_quitte_l_entreprise
df_encoded['a_quitte_l_entreprise'] = df_encoded['a_quitte_l_entreprise'].apply(lambda x: 0 if x == 'Non' else 1)

encoder = ce.BinaryEncoder(cols=['domaine_etude'])
df_encoded = encoder.fit_transform(df_encoded)


df_encoded = df_encoded.drop(columns=['eval_number', 
                                      'augementation_salaire_precedente', 
                                      'distance_domicile_travail_bin', 
                                      'revenu_mensuel_bin', 
                                      'age_bin', 
                                      'code_sondage',
                                      'a_quitte_l_entreprise_enc'], errors='ignore')




cols_quantitative = ['id_employee', 'age', 'revenu_mensuel', 'annees_dans_l_entreprise',
                     'distance_domicile_travail', 'annees_depuis_la_derniere_promotion',
                     'augmentation_salaire_precedente_pourcentage', 'a_quitte_l_entreprise'
                    ]

#------------------ FEATURE ENGINEERING ---------------------
mask_reconaissance = ((df_encoded['heure_supplementaires'])
    & (df_encoded['augmentation_salaire_precedente_pourcentage'] < np.quantile(df_encoded['augmentation_salaire_precedente_pourcentage'], [.5][0])))

mask_deplacement = ((df_encoded['frequence_deplacement'] > 0)
    & (df_encoded['satisfaction_employee_equilibre_pro_perso'] < 3))

mask_nouveaux = ((df_encoded['niveau_hierarchique_poste'] < 4)
    & (df_encoded["annees_dans_l_entreprise"] < 3))

mask_sad = ((df_encoded['poste'].isin([259, 83, 52]))
    & (df_encoded['satisfaction_employee_nature_travail'] < 3))

mask_jeunes = ((df_encoded['age'] > 26) & (df_encoded['age'] < 35)
    & (df_encoded['niveau_education'] > 2))

df_encoded = create_new_feature(df_encoded, mask_reconaissance, "reconnaissance_travail")
df_encoded = create_new_feature(df_encoded, mask_deplacement, "pro_perso_deplacement")
df_encoded = create_new_feature(df_encoded, mask_nouveaux, "nouveaux_employee")
df_encoded = create_new_feature(df_encoded, mask_sad, "non_satisfait")
df_encoded = create_new_feature(df_encoded, mask_jeunes, "jeunes_employee")

#------------------- Parameters ML -------------------

X = df_encoded.drop(columns=[FEATURE_TARGET])
y = df_encoded[FEATURE_TARGET]

models_test = [["Logistic Regression", LogisticRegression(max_iter=10000, class_weight='balanced')],
               ["Catboost", CatBoostClassifier(iterations=2, depth=4)],
               ["LinearSVC", LinearSVC(class_weight='balanced')],
               ["RandomForest", RandomForestClassifier(class_weight='balanced')],
               ["SVC", SVC(kernel='poly')],
               ["SVCKernel", SVC(kernel="rbf", gamma=.1)]
              ]

calibrator = CalibratedClassifierCV(
        LinearSVC(class_weight='balanced'),
        method='sigmoid',
        cv=5
    )

# On n'encode pas la valeur poste dans cette dataframe mais plus tard dans le modele
# Ceci afin de séparer l'encodage entre train et test et éviter l'overfit
# Mais on en a besoin ici donc on va encoder dans une nouvelle df

__all__ = ['df_encoded', 'df_final', 'getEmployee', 'models_test']