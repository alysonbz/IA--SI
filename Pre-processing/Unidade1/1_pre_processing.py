from fontTools.subset import subset
from src.utils import load_volunteer_dataset

volunteer = load_volunteer_dataset()

# Mostre a dimensão do dataset volunteer
print(volunteer.shape)

#mostre os tipos de dados existentes no dataset
print(volunteer.info())

#mostre quantos elementos do dataset estão faltando na coluna
print(volunteer['locality'].isnull().sum())

# Exclua as colunas Latitude e Longitude de volunteer
<<<<<<< HEAD
volunteer_cols = volunteer.drop(['Longitude', 'Latitude'],axis=1)

# Exclua as linhas com valores null da coluna category_desc de volunteer_cols
volunteer_subset = volunteer_cols.dropna(subset='category_desc')

# Print o shape do subset
print(volunteer_subset.shape)
=======
volunteer_cols = volunteer.drop(['Latitude', 'Longitude'],axis =1)

# Exclua as linhas com valores null da coluna category_desc de volunteer_cols
volunteer_subset = volunteer_cols.dropna(subset = 'category_desc')

# Print o shape do subset
print(volunteer.subset.shape)

>>>>>>> 1239a00c96cd4d3adea696c64633d52b04d5adf1

