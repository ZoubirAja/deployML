from database import get_connection
import pandas as pd

from pydantic import BaseModel, Field, field_validator
from typing import Optional
import pandas as pd
import joblib

# ============================================================
# SCHÉMA DE DONNÉES
# ============================================================

FREQUENCE_MAP = {
    "jamais": 0,
    "occasionnel": 1,
    "frequent": 2
}

_, __, TARGET_ENCODING, ___ = joblib.load('model.pkl')
POSTES_VALIDES = set(TARGET_ENCODING.keys())

class EmployeeInput(BaseModel):
    # -------- Obligatoires --------
    heure_supplementaires: int              = Field(..., ge=0, le=1)
    age: int                                = Field(..., ge=18, le=100)
    genre: int                              = Field(..., ge=0, le=1)
    revenu_mensuel: int                     = Field(..., ge=0)
    poste: str                              = Field(..., description="Nom du poste ex: 'manager'")
    nombre_experiences_precedentes: int     = Field(..., ge=0)
    annee_experience_totale: int            = Field(..., ge=0)
    annees_dans_l_entreprise: int           = Field(..., ge=0)
    annees_dans_le_poste_actuel: int        = Field(..., ge=0)
    nombre_participation_pee: int           = Field(..., ge=0)
    nb_formations_suivies: int              = Field(..., ge=0)
    distance_domicile_travail: int          = Field(..., ge=0)
    niveau_education: int                   = Field(..., ge=1, le=5)
    frequence_deplacement: str              = Field(..., description="'jamais', 'occasionnel' ou 'frequent'")
    annees_depuis_la_derniere_promotion: int = Field(..., ge=0)
    annes_sous_responsable_actuel: int      = Field(..., ge=0)
    departement: str                        = Field(..., description="'consulting', 'rh' ou 'commercial'")
    augmentation_salaire_precedente_pourcentage: int = Field(..., ge=0, le=100)

    # -------- Optionnelles (1 à 4) --------
    satisfaction_employee_environnement: Optional[int]          = Field(None, ge=1, le=4)
    note_evaluation_precedente: Optional[int]                   = Field(None, ge=1, le=4)
    niveau_hierarchique_poste: Optional[int]                    = Field(None, ge=1, le=5)
    satisfaction_employee_nature_travail: Optional[int]         = Field(None, ge=1, le=4)
    satisfaction_employee_equipe: Optional[int]                 = Field(None, ge=1, le=4)
    satisfaction_employee_equilibre_pro_perso: Optional[int]    = Field(None, ge=1, le=4)
    note_evaluation_actuelle: Optional[int]                     = Field(None, ge=1, le=4)
    statut_marital: Optional[int]                               = Field(None, ge=0, le=2)
    domaine_etude_0: Optional[int]                              = Field(None, ge=0, le=1)
    domaine_etude_1: Optional[int]                              = Field(None, ge=0, le=1)
    domaine_etude_2: Optional[int]                              = Field(None, ge=0, le=1)

    @field_validator('frequence_deplacement')
    def valider_frequence(cls, v):
        if v.lower() not in FREQUENCE_MAP:
            raise ValueError("frequence_deplacement doit être 'jamais', 'occasionnel' ou 'frequent'")
        return v.lower()

    @field_validator('departement')
    def valider_departement(cls, v):
        if v.lower() not in ['consulting', 'rh', 'commercial']:
            raise ValueError("departement doit être 'consulting', 'rh' ou 'commercial'")
        return v.lower()

    @field_validator('poste')
    def valider_poste(cls, v):
        if v.lower() not in POSTES_VALIDES:
            raise ValueError(f"Poste inconnu. Postes disponibles : {sorted(POSTES_VALIDES)}")
        return v.lower()


SEUIL_AUGMENTATION = 14 # On aura besoin de cette valeur pour l'une des nouvelles feature

def prepare_dataframe(data: EmployeeInput) -> pd.DataFrame:
    """Transforme l'input en DataFrame prêt pour le pipeline."""

    frequence = FREQUENCE_MAP[data.frequence_deplacement]
    dept = data.departement.lower()
    departement_Consulting          = 1 if dept == "consulting" else 0
    departement_Ressources_Humaines = 1 if dept == "rh" else 0
    poste_encoded = TARGET_ENCODING.get(data.poste.lower(), 0.5)
    POSTES_SAD = {"consultant", "representant commercial", "ressources humaines"}

    # -------- Valeurs de base --------
    heure_sup       = data.heure_supplementaires
    augmentation    = data.augmentation_salaire_precedente_pourcentage
    freq_dep        = frequence
    satisfaction_eq = data.satisfaction_employee_equilibre_pro_perso or 3
    niveau_h        = data.niveau_hierarchique_poste or 2
    annees_entrep   = data.annees_dans_l_entreprise
    age             = data.age
    niveau_edu      = data.niveau_education
    nature_travail  = data.satisfaction_employee_nature_travail or 3

    # -------- Feature Engineering (reproduit depuis le notebook) --------
    reconnaissance_travail  = int(heure_sup == 1 and augmentation < SEUIL_AUGMENTATION)
    pro_perso_deplacement   = int(freq_dep > 0 and satisfaction_eq < 3)
    nouveaux_employee       = int(niveau_h < 4 and annees_entrep < 3)
    non_satisfait           = int(data.poste.lower() in POSTES_SAD and nature_travail < 3)
    jeunes_employee         = int(26 < age < 35 and niveau_edu > 2)

    row = {
        "age":                                          age,
        "genre":                                        data.genre,
        "revenu_mensuel":                               data.revenu_mensuel,
        "poste":                                        poste_encoded,
        "nombre_experiences_precedentes":               data.nombre_experiences_precedentes,
        "annee_experience_totale":                      data.annee_experience_totale,
        "annees_dans_l_entreprise":                     annees_entrep,
        "annees_dans_le_poste_actuel":                  data.annees_dans_le_poste_actuel,
        "nombre_participation_pee":                     data.nombre_participation_pee,
        "nb_formations_suivies":                        data.nb_formations_suivies,
        "distance_domicile_travail":                    data.distance_domicile_travail,
        "niveau_education":                             niveau_edu,
        "frequence_deplacement":                        freq_dep,
        "annees_depuis_la_derniere_promotion":          data.annees_depuis_la_derniere_promotion,
        "annes_sous_responsable_actuel":                data.annes_sous_responsable_actuel,
        "heure_supplementaires":                        heure_sup,
        "departement_Consulting":                       departement_Consulting,
        "departement_Ressources Humaines":              departement_Ressources_Humaines,
        "augmentation_salaire_precedente_pourcentage":  augmentation,
        "satisfaction_employee_environnement":          data.satisfaction_employee_environnement or 3,
        "note_evaluation_precedente":                   data.note_evaluation_precedente or 3,
        "niveau_hierarchique_poste":                    niveau_h,
        "satisfaction_employee_nature_travail":         nature_travail,
        "satisfaction_employee_equipe":                 data.satisfaction_employee_equipe or 3,
        "satisfaction_employee_equilibre_pro_perso":    satisfaction_eq,
        "note_evaluation_actuelle":                     data.note_evaluation_actuelle or 3,
        "statut_marital":                               data.statut_marital or 0,
        "domaine_etude_0":                              data.domaine_etude_0 or 0,
        "domaine_etude_1":                              data.domaine_etude_1 or 0,
        "domaine_etude_2":                              data.domaine_etude_2 or 0,
        # Features engineered
        "reconnaissance_travail":                       reconnaissance_travail,
        "pro_perso_deplacement":                        pro_perso_deplacement,
        "nouveaux_employee":                            nouveaux_employee,
        "non_satisfait":                                non_satisfait,
        "jeunes_employee":                              jeunes_employee,
        # id_employee supprimé comme convenu
    }

    return pd.DataFrame([row])

def get_employee(id):
    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute("SELECT * FROM employees WHERE id_employee = %s;", (id,))
            row = cursor.fetchone()
            if row is None:
                return None
            
            columns = [desc[0] for desc in cursor.description] # On récupre les columns name
            return pd.DataFrame([row], columns=columns)


def get_employees_groupe(poste):
    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute("SELECT * FROM employees WHERE poste = %s;", (poste,))  # ← poste au lieu de id
            rows = cursor.fetchall()  # ← fetchall pour récupérer tous les employés du groupe
            if not rows:
                return None
            
            columns = [desc[0] for desc in cursor.description]
            return pd.DataFrame(rows, columns=columns)  # ← rows au lieu de [row]