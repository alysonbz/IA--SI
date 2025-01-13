from src.utils import load_volunteer_dataset
from sklearn.model_selection import  train_test_split
from unicodedata import category

volunteer = load_volunteer_dataset()

# Exclua as colunas Latitude e Longitude de volunteer
volunteer_new = volunteer(['Latitude', 'Longitude'].drop)

# Exclua as linhas com valores null da coluna category_desc de volunteer_new
volunteer = volunteer_new.dropna(['category_desc'])

# mostre o balanceamento das classes em 'category_desc'
print(__,'\n','\n')

# Crie um DataFrame com todas as colunas, com exceção de ``category_desc``
X = __

# Crie um dataframe de labels com a coluna category_desc
y =__

# # Utiliza a amostragem stratificada para separar o dataset em treino e teste
X_train, X_test, y_train, y_test = ___(__, __, stratify=__, random_state=42)

# mostre o balanceamento das classes em 'category_desc' novamente
print(___)