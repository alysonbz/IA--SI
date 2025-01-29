from pandas.conftest import axis_1

from src.utils import load_volunteer_dataset

volunteer = load_volunteer_dataset()

# Mostre a dimensão do dataset volunteer
print(volunteer.shape)

#mostre os tipos de dados existentes no dataset
print(volunteer.info())

#mostre quantos elementos do dataset estão faltando na coluna
<<<<<<< HEAD
print(volunteer["locality"].insull().sum())

# Exclua as colunas Latitude e Longitude de volunteer
volunteer_cols = volunteer.drop("Locality"["Longitude","Latitude"],axis_1)

# Exclua as linhas com valores null da coluna category_desc de volunteer_cols
#volunteer_subset =
print(volunteer.dropna())
# Print o shape do subset
#___
=======
print(volunteer['Locality'].insull().sum())

# Exclua as colunas Latitude e Longitude de volunteer
volunteer_cols = volunteer.drop(['Longitude','Latitude'], axis = 1)

# Exclua as linhas com valores null da coluna category_desc de volunteer_cols
volunteer_subset = volunteer_cols.dropna(subset = 'category_desc')

# Print o shape do subset
print(volunteer.subset.shape)

>>>>>>> 19ff1d64c8fb77cf33e1c8859ac796054638046f


