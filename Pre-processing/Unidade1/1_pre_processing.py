from src.utils import load_volunteer_dataset

volunteer = load_volunteer_dataset()

# Mostre a dimensão do dataset volunteer
print(volunteer.describe())

print("____________________________________________________________")

#mostre os tipos de dados existentes no dataset
print(volunteer.info())

print("____________________________________________________________")


#mostre quantos elementos do dataset estão faltando na coluna
print(volunteer['locality'].isna().sum())

print("____________________________________________________________")



# Exclua as colunas Latitude e Longitude de volunteer
volunteer_cols = volunteer.drop(["Latitude", "Longitude"], axis=1)
print(volunteer_cols)

print("____________________________________________________________")



# Exclua as linhas com valores null da coluna category_desc de volunteer_cols
volunteer_subset = volunteer_cols.dropna(subset=["category_desc"])
print(volunteer_subset)

print("____________________________________________________________")

# Print o shape do subset
print(volunteer_subset.shape)


