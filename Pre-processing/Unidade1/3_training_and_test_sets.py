from pandas.conftest import axis_1

from src.utils import load_volunteer_dataset
_____

volunteer = load_volunteer_dataset()

# Exclua as colunas Latitude e Longitude de volunteer
volunteer_new = volunteer.drop(['Latitude','Longitude'],axis=1)

# Exclua as linhas com valores null da coluna category_desc de volunteer_new
volunteer = volunteer.new.dropna(subset=['category_desc'])

# mostre o balanceamento das classes em 'category_desc'
print(volunteer['category_desc'].value_counts(),'\n','\n')

# Crie um DataFrame com todas as colunas, com exceção de ``category_desc``
X = volunteer.drop(subset=['category_desc'])

# Crie um dataframe de labels com a coluna category_desc
y = __[['__']]

# # Utiliza a a amostragem stratificada para separar o dataset em treino e teste
X_train, X_test, y_train, y_test = __(__, __, stratify=__, random_state=42)

# mostre o balanceamento das classes em 'category_desc' novamente
___