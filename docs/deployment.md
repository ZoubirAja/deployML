# Déploiement

## Local (développement)

```bash
cp .env.example .env.dev
docker compose up --build
docker compose exec api python init_db.py
docker compose exec api python train.py
```

## Production (Hugging Face + Neon)

1. Créer une BDD sur neon.tech
2. Ajouter `DATABASE_URL` dans les secrets HF Space
3. Pusher sur la branche connectée au Space

## Réentraînement

```bash
docker compose exec api python train.py
```

Le nouveau modèle est automatiquement uploadé sur MinIO 
et disponible au prochain démarrage de l'API.