# Base de données

## Schéma

### Table `employees`
Contient les données RH encodées.

| Colonne | Type | Description |
|---|---|---|
| id_employee | INTEGER PK | Identifiant unique |
| age | INTEGER | Age de l'employé |
| poste | FLOAT | Poste encodé (target encoding) |
| ... | ... | ... |

### Table `prediction_logs`
Trace chaque appel à l'API.

| Colonne | Type | Description |
|---|---|---|
| id | INTEGER PK | Auto-increment |
| timestamp | DATETIME | Date/heure de la prédiction |
| id_employee | INTEGER FK | Référence employees (nullable) |
| prediction | INTEGER | 0 ou 1 |
| probabilite_depart | FLOAT | Entre 0 et 1 |
| resultat | TEXT | Texte du résultat |

## Relations

`prediction_logs.id_employee` → `employees.id_employee` (FK nullable)