from src.utils import load_volunteer_dataset

volunteer = load_volunteer_dataset()

# Print os primeiros elementos da coluna hits
print(volunteer['hits'].head())

print("____________________________________________________________")

# Print as caracteristicas da coluna hits
print(volunteer['hits'].info)

print("____________________________________________________________")

# Converta a coluna hits para o tipo int
volunteer['hits'] = volunteer['hits'].astype('int')

print("____________________________________________________________")

# Print as caracteristicas da coluna hits novamente
print(volunteer['hits'].info)

