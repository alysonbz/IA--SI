from src.utils import load_volunteer_dataset

volunteer = load_volunteer_dataset()

# Mostre a dimensão do dataset volunteer
print(volunteer.shape)

#mostre os tipos de dados existentes no dataset
print(volunteer.info())

#mostre quantos elementos do dataset estão faltando na coluna
print(volunteer['locality'] .isna() .sun())

# Exclua as colunas Latitude e Longitude de volunteer
volunteer_cols = volunteer.dorp(['latitude','longitude'] , axis=1)

# Exclua as linhas com valores null da coluna category_desc de volunteer_cols
volunteer_subset = volunteer_cols.dropna(subset = ['category_desc'])

# Print o shape do subset
print(volunteer_subset.shape)


