FEATURE_TARGET = 'a_quitte_l_entreprise'

# Affiche le taux démission globale et le taux de démission selon le masque entré
# La fonction nous affiche aussi le nombre d'employé concernés
def test_new_feature(dataframe, mask):
    feature = dataframe[mask]
    taux_demission_feature = feature['a_quitte_l_entreprise'].mean()
    
    print(f"Taux de démission dans la nouvelle feature : {taux_demission_feature:.2%}")
    print(f"Nombre d'individus dans la feature : {len(feature)}")

# Retourne un df avec une nouvelle feature binaire
# 1 si les conditions du masque sont remplies
def create_new_feature(dataframe, mask, feature_name):
    dataframe[feature_name] = 0
    dataframe.loc[mask, feature_name] = 1

    return dataframe