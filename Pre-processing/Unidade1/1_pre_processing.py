from src.utils import load_volunteer_dataset

volunteer = load_volunteer_dataset()


print("\n* --># Mostre a dimensão do dataset volunteer:\n")
print(volunteer.shape )


print("\n* --> #mostre os tipos de dados existentes no dataset:\n")
print( volunteer.info())


print("\n* --> #mostre quantos elementos do dataset estão faltando na coluna:\n")
print(volunteer ["locality"] .isna() .sum())


print("\n* --> #Exclua as colunas Latitude e Longitude de volunteer:\n")
volunteer_cols = volunteer.drop(["Latitude","Longitude"], axis=1)
print(volunteer_cols)

print("\n* --> #Exclua as linhas com valores null da coluna category_desc de volunteer_cols:\n")
volunteer_subset = volunteer_cols.dropna(subset=["category_desc"])
print(volunteer_subset)

print("\n* --># Print o shape do subset:\n")
print(volunteer_subset.shape)


