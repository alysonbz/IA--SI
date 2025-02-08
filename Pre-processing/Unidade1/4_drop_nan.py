from src.utils import load_volunteer_dataset

volunteer = load_volunteer_dataset()

<<<<<<< HEAD
#Elabore um código para motrar o shape e dados do dataset volunteer corrigido, sem NAN em suas colunas.
#pronto para utilizar em um proxima etapa.
#Contagem de linhas do dataframe original
print(volunteer.shape)

#Checkagem da linhas nulas por coluna
print(volunteer.isnull().sum())

#Deletar colonas com quantidades absurdas de linhas nulas
volunteer_drop = volunteer.drop(['is_priority', 'amsl', 'amsl_unit', 'primary_loc'], axis=1)

#Limpando o resto do dataframe
volunteer_clean = volunteer_drop.dropna()
print(volunteer_clean.shape)
print(volunteer_clean.isnull().sum())
=======
## realize print do dataset volunteer corrigido sem nenhum NAN, para isto removam as colunas NAN e depois as linhas e crie
#um dataframe novo e print este mostrando a contagem de colunas NAN existentes e mostre também o shape novo.
>>>>>>> 8dbee5f0bdad0e083bc03654e1a4101bf868fd0d
