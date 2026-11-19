from src.utils import load_volunteer_dataset

volunteer = load_volunteer_dataset()

# Mostre a dimensão do dataset volunteer
print((volunteer.shape), "\n")

#mostre os tipos de dados existentes no dataset
print(volunteer.info(), "\n")

#mostre quantos elementos do dataset estão faltando na coluna
print(volunteer['locality'].isna().sum(), "\n")


# Exclua as colunas Latitude e Longitude de volunteer
print(volunteer.drop('Latitude', axis = 1).drop('Longitude', axis = 1))

# Exclua as linhas com valores null da coluna category_desc de volunteer_cols
volunteer_cols = volunteer.dropna(subset=['category_desc'])

# Print o shape do subset
volunteer_subset = volunteer['category_desc']
print(volunteer_subset.shape, "\n")
print(volunteer_subset.info(), "\n")