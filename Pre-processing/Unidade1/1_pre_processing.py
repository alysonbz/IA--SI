from src.utils import load_volunteer_dataset

volunteer = load_volunteer_dataset()

# Mostre a dimensão do dataset volunteer
print(voluenteer.shape)

#mostre os tipos de dados existentes no dataset
print(voluenteer.info())

#mostre quantos elementos do dataset estão faltando na coluna
print(voluenteer['locality'].isnul().sum())

# Exclua as colunas Latitude e Longitude de volunteer
volunteer_cols = volunteer.drop(['latitude', 'Longitude'], axis=1)

# Exclua as linhas com valores null da coluna category_desc de volunteer_cols
volunteer_subset = volunteer_cols.dropna(subset='category_desc')

# Print o shape do subset
print(volunteer_subset.shape)


