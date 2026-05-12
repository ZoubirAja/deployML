# API Reference

## Authentification

Toutes les routes protégées nécessitent un header :  
API_KEY=une_cle_secrete_longue_et_random

## Endpoints

### POST /predict/{id_employee}
Prédit le risque de départ d'un employé depuis la BDD.

**Réponse :**
```json
{
  "prediction": 1,
  "probabilite_de_depart": "72%",
  "resultat": "Va quitter l'entreprise"
}
```

### POST /predict_nouveau
Prédit depuis des données brutes.  

**Appel :**
```json
{
  "heure_supplementaires": 1,
  "age": 18,
  "genre": 1,
  "revenu_mensuel": 0,
  "poste": "string",
  "nombre_experiences_precedentes": 0,
  "annee_experience_totale": 0,
  "annees_dans_l_entreprise": 0,
  "annees_dans_le_poste_actuel": 0,
  "nombre_participation_pee": 0,
  "nb_formations_suivies": 0,
  "distance_domicile_travail": 0,
  "niveau_education": 1,
  "frequence_deplacement": "string",
  "annees_depuis_la_derniere_promotion": 0,
  "annes_sous_responsable_actuel": 0,
  "departement": "string",
  "augmentation_salaire_precedente_pourcentage": 100,
  "satisfaction_employee_environnement": 1,
  "note_evaluation_precedente": 1,
  "niveau_hierarchique_poste": 1,
  "satisfaction_employee_nature_travail": 1,
  "satisfaction_employee_equipe": 1,
  "satisfaction_employee_equilibre_pro_perso": 1,
  "note_evaluation_actuelle": 1,
  "statut_marital": 2,
  "domaine_etude_0": 1,
  "domaine_etude_1": 1,
  "domaine_etude_2": 1
}

**Réponse :**
```json
{
  "prediction": 1,
  "probabilite_de_depart": "72%",
  "resultat": "Va quitter l'entreprise"
}

### POST /predict/group/{poste}
Analyse tous les employés d'un poste.

**Réponse :**
```json
{
  "poste": "manager",
  "nombre_employes": 10,
  "taux_de_depart_prevu": "30%",
  "top5_employes_a_risque": [...]
}
```