from src.utils import load_volunteer_dataset

volunteer = load_volunteer_dataset()

#Elabore um código para motrar o shape e dados do dataset volunteer corrigido, sem NAN em suas colunas.
#pronto para utilizar em um proxima etapa.
print('Dimensão do inicio: ', volunteer.shape)

print(volunteer.isnull().sum())

volunteer_coluna = volunteer.drop(['amsl', 'amsl_unit', 'is_priority', 'primary_loc'],axis=1)
print(volunteer_coluna.isnull().sum())

volunteer_line = volunteer_coluna.dropna()
print(volunteer_line.isnull().sum())
print(f'A dimensão final fica: ', volunteer_line.shape)