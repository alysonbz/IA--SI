from src.utils import load_volunteer_dataset

volunteer = load_volunteer_dataset()
print(volunteer)
## realize print do dataset volunteer corrigido sem nenhum NAN, para isto removam as colunas NAN e depois as linhas e crie
#um dataframe novo e print este mostrando a contagem de colunas NAN existentes e mostre também o shape novo.
volunteern=volunteer.dropna(axis=1)
volunteern= volunteern.dropna(axis=0)
print(volunteern.isna().sum())
print(volunteern.shape)
