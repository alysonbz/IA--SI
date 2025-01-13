from src.utils import load_volunteer_dataset

volunteer = load_volunteer_dataset()

# Print os primeiros elementos da coluna hits
print(volunteer['hits'].head())

# Print as caracteristicas da coluna hits
print(volunteer['hits'].info())


# Converta a coluna hits para o tipo int
volunteer_collunType = volunteer['hits'].astype('int64')


# Print as caracteristicas da coluna hits novamente
print(volunteer_collunType.info())