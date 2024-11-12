from src.utils import load_volunteer_dataset

volunteer = load_volunteer_dataset()

# Print os primeiros elementos da coluna hits
print(volunteer['hits'].head())

# Print as caracteristicas da coluna hits
print(volunteer['hits'].describe())

# Converta a coluna hits para o tipo int32
volunteer['hits'] = volunteer['hits'].fillna(0).astype(int)
print(volunteer.dtypes)

# Print novamente as caracteristucas da coluna hits
print(volunteer['hits'].describe())
