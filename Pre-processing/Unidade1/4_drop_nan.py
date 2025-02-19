from src.utils import load_volunteer_dataset

volunteer = load_volunteer_dataset()

#Elabore um código para motrar o shape e dados do dataset volunteer corrigido, sem NAN em suas colunas.
#pronto para utilizar em um proxima etapa.

print("DIMENSÃO: ", volunteer.shape)

print("\nSOMA NaN COLUNAS")
print(volunteer.isnull().sum())

print("\nEXCLUINDO COLUNAS")
volunteer_cols = volunteer.drop(['is_priority', 'amsl', 'amsl_unit', 'primary_loc'], axis=1)
print(volunteer_cols.isnull().sum())

volunteer_line = volunteer_cols.dropna()
print(volunteer_line.isnull().sum())

print("DIMENSÃO FINAL: ", volunteer_line.shape)