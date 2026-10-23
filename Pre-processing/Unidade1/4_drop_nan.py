from src.utils import load_volunteer_dataset

volunteer = load_volunteer_dataset()

## realize print do dataset volunteer corrigido sem nenhum NAN, para isto removam as colunas NAN e depois as linhas e crie
#um dataframe novo e print este mostrando a contagem de colunas NAN existentes e mostre também o shape novo.

# Remova as colunas que possuem valores NAN
volunteer_new = volunteer.dropna(axis=1)

# Remova as linhas que possuem valores NAN
volunteer_new = volunteer_new.dropna(axis=0)

# Mostre a quantidade de valores NAN existentes
print(volunteer_new.isna().sum().sum())

# Mostre o dataset corrigido
print(volunteer_new)

# Mostre o novo shape
print(volunteer_new.shape)