from src.utils import load_volunteer_dataset

volunteer = load_volunteer_dataset()

## realize print do dataset volunteer corrigido sem nenhum NAN, para isto removam as colunas NAN e depois as linhas e crie
#um dataframe novo e print este mostrando a contagem de colunas NAN existentes e mostre também o shape novo.
new_volunteer = volunteer.dropna( axis=1)
new_volunteer = new_volunteer.dropna( axis=0)
print("Contagem de colunas NaN existentes:")
print(new_volunteer.isna().sum())
print("dataframe novo shape:")
print(new_volunteer.shape)
print("Novo datafreme volunteer:")
print(new_volunteer)