from unicodedata import category
from src.utils import load_volunteer_dataset

# Função para dividir o dataset em treino e teste
def train_test_split(X, y, test_size):
    from sklearn.model_selection import train_test_split as tts
    X_train, X_test, y_train, y_test = tts(X, y, test_size=test_size, random_state=42)
    return X_train, X_test, y_train, y_test

volunteer = load_volunteer_dataset()

# Exclua as colunas Latitude e Longitude de volunteer
volunteer_new = volunteer.drop(columns=['Latitude', 'Longitude'])

# Exclua as linhas com valores null da coluna category_desc de volunteer_new
volunteer = volunteer_new.dropna(subset=['category_desc'])

# Mostre o balanceamento das classes em 'category_desc'
print(volunteer['category_desc'].value_counts(), '\n\n')

# Crie um DataFrame com todas as colunas, com exceção de 'category_desc'
X = volunteer.drop(columns=['category_desc'])

# Crie um dataframe de labels com a coluna category_desc
y = volunteer[['category_desc']]

# Utiliza a amostragem estratificada para separar o dataset em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Mostre o balanceamento das classes em 'category_desc' novamente
print(y_train['category_desc'].value_counts(), '\n\n', y_test['category_desc'].value_counts())
