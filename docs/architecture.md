# Architecture

## Vue globale

Données CSV → Preprocessing → LinearSVC → Calibrateur → model.pkl  
↓  
RH → API FastAPI → PostgreSQL (logs)                      MinIO

## Les 4 containers Docker

| Service | Rôle |
|---|---|
| api | FastAPI + uvicorn |
| db | PostgreSQL |
| minio | Stockage model.pkl |
| scheduler | Réentraînement automatique |

## Séparation train / serve

Le modèle est entraîné séparément via `train.py` 
et sauvegardé sur MinIO. L'API télécharge le modèle 
au démarrage sans jamais réentraîner.