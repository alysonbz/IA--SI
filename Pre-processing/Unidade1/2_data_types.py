from pandas.conftest import axis_1

from src.utils import load_volunteer_dataset

volunteer = load_volunteer_dataset()

# Print os primeiros elementos da coluna hits
print(volunteer['hits'].head())

# Print as caracteristicas da coluna hits
print(volunteer['hits'].describe())

# Converta a coluna hits para o tipo int
volunteer['hits'] = volunteer['hits'].fillna(0).astype(int)
print(volunteer.dtypes)
<<<<<<< HEAD
# Print as caracteristicas da coluna hits novamente
print(volunteer['hits'].describe())
=======

# Print as caracteristicas da coluna hits novamente
print(volunteer['hits'].describe())
>>>>>>> 19ff1d64c8fb77cf33e1c8859ac796054638046f
