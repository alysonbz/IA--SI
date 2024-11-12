from src.utils import load_volunteer_dataset
from sklearn.model_selection import  train_test_split

volunteer = load_volunteer_dataset()

# Exclua as colunas Latitude e Longitude de volunteer
volunteer_new = ___

# Exclua as linhas com valores null da coluna category_desc de volunteer_new
volunteer = __

# mostre o balanceamento das classes em 'category_desc'
print(__,'\n','\n')

# Crie um DataFrame com todas as colunas, com exceção de ``category_desc``
X = __

# Crie um dataframe de labels com a coluna category_desc
y =__

# # Utiliza a a amostragem stratificada para separar o dataset em treino e teste
X_train, X_test, y_train, y_test = ___(__, __, stratify=__, random_state=42)

# mostre o balanceamento das classes em 'category_desc' novamente
print(___)