from src.utils import load_volunteer_dataset

volunteer = load_volunteer_dataset()

# Mostre a dimensão do dataset volunteer
print("dimenção data set volunteer")
print(volunteer.shape,"\n")

#mostre os tipos de dados existentes no dataset
print(volunteer.info(),"\n")

#mostre quantos elementos do dataset estão faltando na coluna
print("quantos elementos do data set estão faltando na coluna quality:")
print(volunteer["locality"].isna().sum(),"\n")

# Exclua as colunas Latitude e Longitude de volunteer
print("remover as colunas latitude e longetude")
volunteer_cols = volunteer.drop(["latitude", "longitude"], axis=1)
print(volunteer_cols, "\n")


# Exclua as linhas com valores null da coluna category_desc de volunteer_cols
print("excluindo as linhas nulas")
volunteer_subset = volunteer_cols.drompna(subset=["category_desc"])
print(volunteer_subset, "\n")

# Print o shape do subset
print("shape do subset")
print(volunteer_subset.shape)
