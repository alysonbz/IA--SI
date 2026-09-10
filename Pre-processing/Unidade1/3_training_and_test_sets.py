from src.utils import load_volunteer_dataset

volunteer = load_volunteer_dataset()

# Exclua as colunas Latitude e Longitude de volunteer
#volunteer_new = __

# Exclua as linhas com valores null da coluna category_desc de volunteer_new
print(volunteer)

# mostre o balanceamento das classes em 'category_desc'
#print(___['category_desc'].__,'\n','\n')

#X = volunteer.__(__, axis=1)

# Crie um dataframe de labels com a coluna category_desc
#y = __[['__']]

#  Utiliza a a amostragem stratificada para separar o dataset em treino e teste
#X_train, X_test, y_train, y_test = __(__, __, stratify=__, random_state=42)

# mostre o balanceamento das classes em 'category_desc' novamente
