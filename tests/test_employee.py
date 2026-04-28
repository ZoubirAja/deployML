from employee import prepare_dataframe, EmployeeInput
import joblib

def test_prepare_dataframe_colonnes(employe_valide):
    """prepare_dataframe doit retourner toutes les colonnes attendues"""
    _, _, _, feature_names, _ = joblib.load('model.pkl')
    donnees = EmployeeInput(**employe_valide)
    df = prepare_dataframe(donnees)

    for col in feature_names:
        assert col in df.columns, f"Colonne manquante : {col}"

def test_prepare_dataframe_sans_nan(employe_valide):
    """prepare_dataframe ne doit pas retourner de NaN"""
    donnees = EmployeeInput(**employe_valide)
    df = prepare_dataframe(donnees)

    assert df.isna().sum().sum() == 0  # aucun NaN

def test_feature_engineering_jeune(employe_valide):
    """Un employé jeune et éduqué doit avoir jeunes_employee = 1"""
    employe_valide["age"] = 30
    employe_valide["niveau_education"] = 4
    donnees = EmployeeInput(**employe_valide)
    df = prepare_dataframe(donnees)

    assert df["jeunes_employee"].iloc[0] == 1