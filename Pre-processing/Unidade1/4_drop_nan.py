from src.utils import load_volunteer_dataset

volunteer = load_volunteer_dataset()

#Elabore um código para motrar o shape e dados do dataset volunteer corrigido, sem NAN em suas colunas.
#pronto para utilizar em um proxima etapa.

print(volunteer.shape)
print(volunteer.isnull().sum())
volunteercorrigido =volunter.drop(['is_priority', 'amsl', 'asml_unit', 'primary_loc'],axis =1)
print(volunteercorrigido)

volunteerfinal = volunteercorrigido.dropna()
print(volunteerfinal)

print(volunteerfinal.isnull().sum())