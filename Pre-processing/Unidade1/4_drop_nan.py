from src.utils import load_volunteer_dataset

volunteer = load_volunteer_dataset()

#Elabore um código para motrar o shape e dados do dataset volunteer corrigido, sem NAN em suas colunas.
print("Dimensão inicial", volunteer.shape)
print(volunteer.isnull().sum())

volunteer_cols = volunteer.drop(['amsl', 'amsl_unit', 'primary_loc', 'is_priority'], axis=1)
print(volunteer_cols.isnull().sum())

volunteer_line = volunteer_cols.dropna()
print(volunteer_line.isnull().sum())

print("Dimensão final", volunteer_line.shape)
#pronto para utilizar em um proxima etapa.