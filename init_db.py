from config import df_encoded
from database import get_connection
from psycopg2.extras import execute_values
import joblib

def init_db():
    with get_connection() as conn:
        with conn.cursor() as cursor:

            # Créer la table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS employees (
                    satisfaction_employee_environnement INTEGER,
                    note_evaluation_precedente INTEGER,
                    niveau_hierarchique_poste INTEGER,
                    satisfaction_employee_nature_travail INTEGER,
                    satisfaction_employee_equipe INTEGER,
                    satisfaction_employee_equilibre_pro_perso INTEGER,
                    note_evaluation_actuelle INTEGER,
                    heure_supplementaires INTEGER,
                    id_employee INTEGER PRIMARY KEY,
                    age INTEGER,
                    genre INTEGER,
                    revenu_mensuel INTEGER,
                    statut_marital INTEGER,
                    poste FLOAT,
                    nombre_experiences_precedentes INTEGER,
                    annee_experience_totale INTEGER,
                    annees_dans_l_entreprise INTEGER,
                    annees_dans_le_poste_actuel INTEGER,
                    a_quitte_l_entreprise INTEGER,
                    nombre_participation_pee INTEGER,
                    nb_formations_suivies INTEGER,
                    distance_domicile_travail INTEGER,
                    niveau_education INTEGER,
                    domaine_etude_0 INTEGER,
                    domaine_etude_1 INTEGER,
                    domaine_etude_2 INTEGER,
                    frequence_deplacement INTEGER,
                    annees_depuis_la_derniere_promotion INTEGER,
                    annes_sous_responsable_actuel INTEGER,
                    departement_consulting BOOLEAN,
                    departement_ressources_humaines BOOLEAN,
                    augmentation_salaire_precedente_pourcentage INTEGER,
                    reconnaissance_travail INTEGER,
                    pro_perso_deplacement INTEGER,
                    nouveaux_employee INTEGER,
                    non_satisfait INTEGER,
                    jeunes_employee INTEGER
                );
            """)

            # Insérer les données du dataframe
            df_clean = df_encoded.rename(columns={
                "departement_Consulting": "departement_consulting",
                "departement_Ressources Humaines": "departement_ressources_humaines"
            })

            _, __, target_encoding, ___ = joblib.load('model.pkl')
            target_encoding = {k.lower(): v for k, v in target_encoding.items()}
            df_clean['poste'] = df_clean['poste'].str.lower().map(target_encoding)


            columns = df_clean.columns.tolist()
            values = [tuple(row) for row in df_clean.itertuples(index=False)]

            execute_values(
                cursor,
                f"INSERT INTO employees ({', '.join(columns)}) VALUES %s ON CONFLICT (id_employee) DO NOTHING;",
                values
            )

            conn.commit()
            print(f"{len(values)} lignes insérées avec succès !")

if __name__ == "__main__":
    init_db()