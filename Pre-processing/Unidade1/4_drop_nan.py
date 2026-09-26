from src.utils import load_volunteer_dataset

volunteer = load_volunteer_dataset()

## realize print do dataset volunteer corrigido sem nenhum NAN, para isto removam as colunas NAN e depois as linhas e crie
#um dataframe novo e print este mostrando a contagem de colunas NAN existentes e mostre também o shape novo.

volunteer_clean = volunteer.dropna(axis=1).dropna(axis=0)

print(volunteer_clean.isna().sum())

print(volunteer_clean.shape)