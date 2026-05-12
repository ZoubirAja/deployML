# DeployML

API de prédiction de départ employé basée sur un modèle LinearSVC.

## Démarrage rapide

```bash
git clone https://github.com/toi/deployml
cd deployml
cp .env.example .env.dev
docker compose up --build
docker compose exec api python init_db.py
docker compose exec api python train.py
```

## Stack technique

| Composant | Outil |
|---|---|
| API | FastAPI |
| ML | scikit-learn, LinearSVC |
| BDD | PostgreSQL + SQLAlchemy |
| Stockage modèle | MinIO |
| Conteneurisation | Docker |
| CI/CD | GitHub Actions |
| Production | Hugging Face + Neon |