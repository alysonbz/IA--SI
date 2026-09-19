from src.utils import load_volunteer_dataset

volunteer = load_volunteer_dataset()

#a partir daqui é meu. e acho q só tira as colunas
print("*--> dataset original:<--*\n",volunteer)
volunteer_novo = volunteer.dropna(axis=1)

print("\n *--> dataframe sem 'NaN':\n",volunteer_novo)

print("\n *-->tamanho do novo dataframe<-- \n", volunteer_novo.shape)

## realize print do dataset volunteer corrigido sem nenhum NAN, para isto removam as colunas NAN e depois as linhas e crie
#um dataframe novo e print este mostrando a contagem de colunas NAN existentes e mostre também o shape novo.