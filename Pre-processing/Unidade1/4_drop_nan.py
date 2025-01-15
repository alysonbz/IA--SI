from src.utils import load_volunteer_dataset

volunteer = load_volunteer_dataset()

<<<<<<< HEAD
#Elabore um código para mostrar o shape e dados do dataset volunteer corrigido, sem NAN em suas colunas.
=======
#Elabore um código para motrar o shape e dados do dataset volunteer corrigido, sem NAN em suas colunas.
>>>>>>> 1239a00c96cd4d3adea696c64633d52b04d5adf1
#pronto para utilizar em um proxima etapa.

print(volunteer.shape)
print(volunteer.isnull().sum())
<<<<<<< HEAD

volunteer_drop = volunteer.drop(['is_priority','amsl','amsl_unit','primary_loc'], axis=1)

volunteer_clean = volunteer_drop.dropna()


print(volunteer_clean.shape)
print(volunteer_clean.isnull() .sum())
=======
volunteercorrigido =volunter.drop(['is_priority', 'amsl', 'asml_unit', 'primary_loc'],axis =1)
print(volunteercorrigido)

volunteerfinal = volunteercorrigido.dropna()
print(volunteerfinal)

print(volunteerfinal.isnull().sum())
>>>>>>> 1239a00c96cd4d3adea696c64633d52b04d5adf1
