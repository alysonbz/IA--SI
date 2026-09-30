from pandas.core.interchange import column

from src.utils import load_volunteer_dataset

volunteer = load_volunteer_dataset()

# Print os primeiros elementos da coluna hits
print(volunteer.head())

# Print as caracteristicas da coluna hits
column_hits = volunteer['hits']
print(column_hits.head())

# Converta a coluna hits para o tipo int
column_hits = volunteer['hits'].astype('int32')
print(column_hits.head())

# Print as caracteristicas da coluna hits novamente
print(column_hits.head())
print(column_hits.shape)
