from src.utils import load_volunteer_dataset

volunteer = load_volunteer_dataset()

<<<<<<< HEAD
#Elabore um código para motrar o shape e dados do dataset volunteer corrigido, sem NAN em suas colunas.
#pronto para utilizar em um proxima etapa.

print('Dimensão iniciial', volunteer.shape)
print(volunteer.isnull().sum())

volunteer_cols = volunteer.drop(['amsl', 'amsl_unit', 'primary_loc', 'is_priority'],axis=1)
print(volunteer_cols.isnull().sum())

volunteer_line = volunteer_cols.dropna()
print(volunteer_line.isnull().sum())

print("Dimensão final", volunteer_line.shape)

#pronto para ser utilizado
=======
## realize print do dataset volunteer corrigido sem nenhum NAN, para isto removam as colunas NAN e depois as linhas e crie
#um dataframe novo e print este mostrando a contagem de colunas NAN existentes e mostre também o shape novo.
>>>>>>> 8dbee5f0bdad0e083bc03654e1a4101bf868fd0d
