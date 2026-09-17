from src.utils import load_volunteer_dataset

volunteer = load_volunteer_dataset()

print("\n* --># Print os primeiros elementos da coluna hits:\n")
print(volunteer["hits"] .head())

print("\n* --># Print as caracteristicas da coluna hits:\n")
print(volunteer["hits"].info())

print("\n* --># Converta a coluna hits para o tipo int32:\n")
volunteer["hits"] = volunteer["hits"].astype("int32")
print(volunteer.dtypes)

print("\n* --># Print as caracteristicas da coluna hits novamente:\n")
print(volunteer["hits"].info())

