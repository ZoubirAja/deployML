import pandas as pd
import matplotlib.pyplot as plt

FEATURE_TARGET = 'a_quitte_l_entreprise'

# Prends une feature quantitative et une feature qualitative pour faire un boxplot
def moustache(category, valeurs, title):
    # Propriétés graphiques (pas très importantes)    
    medianprops = {'color':"black"}
    meanprops = {'marker':'o', 'markeredgecolor':'black',
                'markerfacecolor':'firebrick'}
    groupes = []
    for sexe in pd.unique(df[category]):
        groupes.append(df[df[category]==sexe][valeurs])
    
    plt.boxplot(groupes, tick_labels=pd.unique(df[category]), showfliers=False, medianprops=medianprops, 
                vert=False, patch_artist=True, showmeans=True, meanprops=meanprops)
    plt.title(title)
    plt.show()

# Retourne un tableau avec 2 colonnes, param_1 regroupe les valeurs par catégories
# param_2 est utilisé pour la moyenne de la 2eme colonne. Si les valeurs sont 0, 1. Cela revient à un pourcentage
def tableau_contengence_moyenne(dataframe, param_1, param_2, title='', show_graph=False):
    mean_param = "moy_" + param_2
    taux = dataframe.groupby(param_1)[param_2].mean().reset_index()
    taux.columns = [param_1, mean_param]

    if show_graph:
        fig, ax = plt.subplots()
        plt.title(title)
        plt.barh(taux[param_1], taux[mean_param], align='center')
        plt.show()
    return taux

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