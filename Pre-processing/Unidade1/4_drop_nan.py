from src.utils import load_volunteer_dataset

volunteer = load_volunteer_dataset()

#Elabore um código para mostrar o shape e dados do dataset volunteer corrigido, sem NAN em suas colunas.
#pronto para utilizar em um proxima etapa.

print(volunteer.shape)
print(volunteer.isnull().sum())

volunteer_drop = volunteer.drop(['is_priority','amsl','amsl_unit','primary_loc'], axis=1)

volunteer_clean = volunteer_drop.dropna()


print(volunteer_clean.shape)
print(volunteer_clean.isnull() .sum())