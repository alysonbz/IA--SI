from src.utils import load_volunteer_dataset

volunteer = load_volunteer_dataset()

#Elabore um código para motrar o shape e dados do dataset volunteer corrigido, sem NAN em suas colunas.
#pronto para utilizar em um proxima etapa.
#print(volunteer.shape)

#print(volunteer.info())


#print(volunteer["locality"].isnull().sum())

volunteer_cols = volunteer.drop(['Longitude','Latitude'],axis=1)

volunteer_subset = volunteer_cols.dropna(subset='category_desc')

#print(volunteer.dropna())
#print(volunteer)
#print(volunteer.isnull().sum())
volunteer_sem = volunteer.drop(['is_priority','amsl','amsl_unit','primary_loc'],axis=1)
volunteerfinal = volunteer_sem.dropna()
print(volunteerfinal.isnull().sum())