# models.py
from sqlalchemy import Column, Integer, Float, Boolean, String, DateTime, Text, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from database import Base

class Employee(Base):
    __tablename__ = "employees"

    id_employee                             = Column(Integer, primary_key=True)
    age                                     = Column(Integer)
    genre                                   = Column(Integer)
    revenu_mensuel                          = Column(Integer)
    poste                                   = Column(Float)
    heure_supplementaires                   = Column(Integer)
    nombre_experiences_precedentes          = Column(Integer)
    annee_experience_totale                 = Column(Integer)
    annees_dans_l_entreprise               = Column(Integer)
    annees_dans_le_poste_actuel            = Column(Integer)
    nombre_participation_pee               = Column(Integer)
    nb_formations_suivies                  = Column(Integer)
    distance_domicile_travail              = Column(Integer)
    niveau_education                       = Column(Integer)
    frequence_deplacement                  = Column(Integer)
    annees_depuis_la_derniere_promotion    = Column(Integer)
    annes_sous_responsable_actuel          = Column(Integer)
    departement_consulting                 = Column(Boolean)
    departement_ressources_humaines        = Column(Boolean)
    augmentation_salaire_precedente_pourcentage = Column(Integer)
    satisfaction_employee_environnement    = Column(Integer)
    note_evaluation_precedente             = Column(Integer)
    niveau_hierarchique_poste              = Column(Integer)
    satisfaction_employee_nature_travail   = Column(Integer)
    satisfaction_employee_equipe           = Column(Integer)
    satisfaction_employee_equilibre_pro_perso = Column(Integer)
    note_evaluation_actuelle               = Column(Integer)
    statut_marital                         = Column(Integer)
    domaine_etude_0                        = Column(Integer)
    domaine_etude_1                        = Column(Integer)
    domaine_etude_2                        = Column(Integer)
    a_quitte_l_entreprise                  = Column(Integer)
    reconnaissance_travail                 = Column(Integer)
    pro_perso_deplacement                  = Column(Integer)
    nouveaux_employee                      = Column(Integer)
    non_satisfait                          = Column(Integer)
    jeunes_employee                        = Column(Integer)

    predictions = relationship("PredictionLog", back_populates="employee")


class PredictionLog(Base):
    __tablename__ = "prediction_logs"

    id                  = Column(Integer, primary_key=True, autoincrement=True)
    timestamp           = Column(DateTime, server_default=func.now())
    id_employee         = Column(Integer, ForeignKey("employees.id_employee"), nullable=True)
    prediction          = Column(Integer)
    probabilite_depart  = Column(Float)
    resultat            = Column(Text)

    employee = relationship("Employee", back_populates="predictions")