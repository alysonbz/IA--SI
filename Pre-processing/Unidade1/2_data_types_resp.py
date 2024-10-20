from src.utils import load_volunteer_dataset

volunteer = load_volunteer_dataset()

# Print os primeiros elementos da coluna hits
print(volunteer["hits"].head())

# Print as caracteristicas da coluna hits
print(volunteer['hits'].dtype)

# Converta a coluna hits para o tipo int32
volunteer["hits"] = volunteer["hits"].astype('int32')

# Print novamente as caracteristucas da coluna hits
print(volunteer["hits"].head())