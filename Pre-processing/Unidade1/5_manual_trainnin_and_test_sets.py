from sklearn.model_selection import train_test_split as sklearn_train_test_split
from src.utils import load_volunteer_dataset

volunteer = load_volunteer_dataset()


def train_test_split(X, y, test_size, random_seed=1):
    # Divisão estratificada utilizando o scikit-learn
    X_train, X_test, y_train, y_test = sklearn_train_test_split(
        X, y, test_size=test_size, random_state=random_seed, stratify=y
    )
    return X_train, X_test, y_train, y_test


# Exclua as colunas Latitude e Longitude de volunteer
volunteer_new = volunteer.drop(columns=['latitude', 'longitude'])

# Exclua as linhas com valores null da coluna category_desc de volunteer_new
volunteer = volunteer_new.dropna(subset=['category_desc'])

# Mostre o balanceamento das classes em 'category_desc'
print(volunteer['category_desc'].value_counts(), '\n', '\n')

# Crie um DataFrame com todas as colunas, com exceção de 'category_desc'
X = volunteer.drop('category_desc', axis=1)

# Crie um dataframe de labels com a coluna category_desc
y = volunteer[['category_desc']]

# Utiliza a amostragem estratificada para separar o dataset em treino e teste
test_size = 0.2
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size, random_seed=1
)

# Mostre o balanceamento das classes em 'category_desc' novamente (no conjunto de treino)
print(y_train['category_desc'].value_counts())