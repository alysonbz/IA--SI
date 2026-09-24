from src.utils import load_volunteer_dataset

volunteer = load_volunteer_dataset()

# Print os primeiros elementos da coluna hits
print("5 primeiros elementos da coluna hits de volunteers")
print(volunteer["hits"].head(), "\n")

# Print as caracteristicas da coluna hits
print("caracteristicas da coluna hits")
print(volunteer["hits"].info(), "\n")

# Converta a coluna hits para o tipo int
volunteer["hits"] = volunteer["hits"].astype("int32")
print("converter hits para tipo 32")
print(volunteer.dtypes, "\n")

# Print as caracteristicas da coluna hits novamente
print("caracteristicas da coluna hits")
print(volunteer["hits"].info, "\n")
