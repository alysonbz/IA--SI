from numpy.testing.print_coercion_tables import print_new_cast_table

from src.utils import load_volunteer_dataset

volunteer = load_volunteer_dataset()

#Elabore um código para motrar o shape e dados do dataset volunteer corrigido, sem NAN em suas colunas.
#pronto para utilizar em um proxima etapa.
print(volunteer.info())
print(volunteer.isnull().sum())
volunteer.will = volunteer.drop(['primary_loc','amsl','amsl_unit','is_priority'],axis=1)
print(volunteer.will)

volunteerfinal = volunteer.dropna()
print(volunteerfinal.isnull().sum())