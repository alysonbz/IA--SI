from src.utils import load_volunteer_dataset

volunteer = load_volunteer_dataset()

# Print os primeiros elementos da coluna hits
print("os 5 primeiros elementos da coluna hits do dataset volunteer:")
print(volunteer["hits"].head(), "\n")

# Print as caracteristicas da coluna hits
print("caracteristicas da coluna hits:")
print(volunteer["hits"].info(), "\n")

# Converta a coluna hits para o tipo int
volunteer["hits"] = volunteer["hits"].astype("int32")
print("converter a coluna hits de volunteer para tipo int32:")
print(volunteer.dtypes, "\n")

# Print as caracteristicas da coluna hits novamente
print("caracteristicas da coluna hits:")
print(volunteer["hits"].info(), "\n")
