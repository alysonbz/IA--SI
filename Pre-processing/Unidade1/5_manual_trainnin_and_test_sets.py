from src.utils import load_volunteer_dataset
import numpy as np

volunteer = load_volunteer_dataset()

def train_test_split(X, y, test_size, random_seed=1):
    # SEU CÓDIGO AQUI
    np.random.seed(random_seed)

    train_indices = []
    test_indices = []

    for classe in y['category_desc'].unique():
        indices = y[y['category_desc'] == classe].index.tolist()

        np.random.shuffle(indices)

        qtd_teste = int(len(indices) * test_size)

        test_indices.extend(indices[:qtd_teste])
        train_indices.extend(indices[qtd_teste:])

    X_train = X.loc[train_indices]
    X_test = X.loc[test_indices]

    y_train = y.loc[train_indices]
    y_test = y.loc[test_indices]
    return X_train, X_test, y_train, y_test


# Exclua as colunas Latitude e Longitude de volunteer
volunteer_new = volunteer.drop(['Latitude', 'Longitude'], axis=1)

# Exclua as linhas com valores null da coluna category_desc de volunteer_new
volunteer = volunteer_new.dropna(subset=['category_desc'])

# mostre o balanceamento das classes em 'category_desc'
print(volunteer['category_desc'].value_counts(), '\n', '\n')

# Crie um DataFrame com todas as colunas, com exceção de category_desc
X = volunteer.drop(['category_desc'], axis=1)

# Crie um dataframe de labels com a coluna category_desc
y = volunteer[['category_desc']]

# Utiliza a amostragem stratificada para separar o dataset em treino e teste
test_size = 0.25
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size, random_seed=1
)

# mostre o balanceamento das classes em 'category_desc' novamente
print(y_train['category_desc'].value_counts(), '\n', '\n')