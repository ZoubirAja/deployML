from database import get_connection
import pandas as pd

def get_employee(id):
    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute("SELECT * FROM employees WHERE id_employee = %s;", (id,))
            row = cursor.fetchone()
            if row is None:
                return None
            
            columns = [desc[0] for desc in cursor.description] # On récupre les columns name
            return pd.DataFrame([row], columns=columns)

# class Employee(SQLModel, table=True):
#     id_employee: int = Field(primary_key=True)
#     satisfaction_employee_environnement: int
#     note_evaluation_precedente: int
#     niveau_hierarchique_poste: int
#     satisfaction_employee_nature_travail: int
#     satisfaction_employee_equipe: int
#     satisfaction_employee_equilibre_pro_perso: int
#     note_evaluation_actuelle: int
#     heure_supplementaires: int
#     age: int
#     genre: int
#     revenu_mensuel: int
#     statut_marital: int
#     poste: int
#     nombre_experiences_precedentes: int
#     annee_experience_totale: int
#     annees_dans_l_entreprise: int
#     annees_dans_le_poste_actuel: int
#     a_quitte_l_entreprise: int
#     nombre_participation_pee: int
#     nb_formations_suivies: int
#     distance_domicile_travail: int
#     niveau_education: int
#     domaine_etude_0: int
#     domaine_etude_1: int
#     domaine_etude_2: int
#     frequence_deplacement: int
#     annees_depuis_la_derniere_promotion: int
#     annes_sous_responsable_actuel: int
#     departement_Consulting: int
#     departement_Ressources_Humaines: int  # espace → underscore
#     augmentation_salaire_precedente_pourcentage: int
#     reconnaissance_travail: int
#     pro_perso_deplacement: int
#     nouveaux_employee: int
#     non_satisfait: int
#     jeunes_employee: int

