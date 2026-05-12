from config import df_encoded
from database import engine, Base, SessionLocal
from models import Employee
import joblib

def init_db():
    print("Création des tables...")
    # Crée toutes les tables définies dans models.py
    Base.metadata.create_all(bind=engine)
    print("Tables créées")

    # Insérer les données
    _, __, target_encoding, ___, ____ = joblib.load('model.pkl')
    target_encoding = {k.lower(): v for k, v in target_encoding.items()}

    df_clean = df_encoded.copy()
    df_clean = df_clean.rename(columns={
        "departement_Consulting": "departement_consulting",
        "departement_Ressources Humaines": "departement_ressources_humaines"
    })
    df_clean['poste'] = df_clean['poste'].str.lower().map(target_encoding)

    db = SessionLocal()
    try:
        # Vérifier si la table est déjà peuplée
        if db.query(Employee).count() > 0:
            print("Table déjà peuplée — skip")
            return

        employees = [
            Employee(**{k: v for k, v in row.items()})
            for row in df_clean.to_dict(orient='records')
        ]
        db.bulk_save_objects(employees)
        db.commit()
        print(f"{len(employees)} employés insérés ")
    except Exception as e:
        db.rollback()
        print(f"Erreur : {e}")
    finally:
        db.close()

if __name__ == "__main__":
    init_db()