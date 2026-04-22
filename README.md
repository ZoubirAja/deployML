# 🧠 Projet MLOps — Prédiction de départ employé

API FastAPI permettant de prédire la probabilité qu'un employé quitte l'entreprise, basée sur un modèle LinearSVC calibré.

---

## 📋 Prérequis

- Python 3.10+
- PostgreSQL installé et en cours d'exécution
- Git

---

## 🚀 Installation

### 1. Cloner le repo

```bash
git clone https://github.com/ton-user/ton-repo.git
cd ton-repo
```

### 2. Créer un environnement virtuel

```bash
python -m venv venv
source venv/bin/activate        # Linux / Mac
venv\Scripts\activate           # Windows
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

---

## ⚙️ Configuration

### 4. Créer le fichier `.env`

Crée un fichier `.env` à la racine du projet :

```bash
cp .env.example .env
```

Puis remplis les valeurs dans `.env` :

```env
DB_HOST=localhost
DB_PORT=5432
DB_NAME=nom_de_ta_bdd
DB_USER=ton_user
DB_PASSWORD=ton_mot_de_passe
```

### 5. Créer la base de données PostgreSQL

```bash
psql -U postgres
```

Dans le shell psql :

```sql
CREATE DATABASE nom_de_ta_bdd;
CREATE USER ton_user WITH PASSWORD 'ton_mot_de_passe';
GRANT ALL PRIVILEGES ON DATABASE nom_de_ta_bdd TO ton_user;
\q
```

---

## 🗄️ Initialisation de la base de données

### 6. Peupler la base de données

```bash
python init_db.py
```

Tu devrais voir :
```
XXX lignes insérées avec succès !
```

---

## 🤖 Entraînement du modèle

### 7. Entraîner et sauvegarder le modèle

```bash
python train.py
```

Tu devrais voir :
```
Modèle entraîné — F1 score : 0.XXX
```

Cela génère un fichier `model.pkl` à la racine du projet.

---

## 🌐 Lancer l'API

### 8. Démarrer le serveur

```bash
fastapi dev main.py         # mode développement
```

L'API est accessible sur : http://localhost:8000

La documentation interactive est accessible sur : http://localhost:8000/docs

---

## 📡 Endpoints disponibles

| Méthode | Route | Description |
|---|---|---|
| `POST` | `/predict/{id_employee}` | Prédiction par ID employé (depuis la BDD) |
| `POST` | `/predict` | Prédiction depuis des données brutes |
| `POST` | `/predict/group/{poste}` | Prédiction pour tout un groupe de poste |

### Exemple d'appel `/predict`

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
```

---

## 📁 Structure du projet

```
.
├── main.py           # API FastAPI — endpoints
├── model.py          # Définition et entraînement du modèle
├── train.py          # Script d'entraînement à lancer manuellement
├── config.py         # Chargement et preprocessing des données
├── employee.py       # Schémas Pydantic, prepare_dataframe, requêtes BDD
├── database.py       # Connexion PostgreSQL
├── init_db.py        # Création et peuplement de la table employees
├── requirements.txt  # Dépendances Python
├── .env              # Variables d'environnement (non commité)
├── .env.example      # Template du .env
├── .gitignore        # Fichiers ignorés par git
└── model.pkl         # Modèle entraîné (non commité)
```

---

## 🔄 Réentraîner le modèle

Si les données changent, relance simplement :

```bash
python train.py
```

Le `model.pkl` sera écrasé et l'API utilisera automatiquement le nouveau modèle au prochain appel.

---

## ⚠️ Fichiers non commités

Ces fichiers ne sont pas sur le repo pour des raisons de sécurité :

- `.env` — contient les credentials de la BDD
- `model.pkl` — généré localement via `train.py`
