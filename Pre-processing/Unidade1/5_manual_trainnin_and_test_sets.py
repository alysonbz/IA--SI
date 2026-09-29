from src.utils import load_volunteer_dataset

volunteer = load_volunteer_dataset()


def train_test_split(X, y, test_size, random_seed=1):
    # SEU CÓDIGO AQUI
    df = X.copy()
    df['_target'] = y

    # Embaralha os dados
    df_shuffled = df.sample(frac=1, random_state=random_seed)

    # Calcula o tamanho do conjunto de teste
    test_count = int(len(df_shuffled) * test_size)

    # Separa em teste e treino
    test_df = df_shuffled.iloc[:test_count]
    train_df = df_shuffled.iloc[test_count:]

    # Separa as features (X) dos rótulos (y)
    X_train = train_df.drop(columns=['_target'])
    X_test = test_df.drop(columns=['_target'])
    y_train = train_df[['_target']].rename(columns={'_target': y.columns[0]})
    y_test = test_df[['_target']].rename(columns={'_target': y.columns[0]})

    return X_train, X_test, y_train, y_test


# Exclua as colunas Latitude e Longitude de volunteer
volunteer_new = volunteer.drop(columns=['Latitude', 'Longitude'])

# Exclua as linhas com valores null da coluna category_desc de volunteer_new
volunteer = volunteer_new.dropna(subset=['category_desc'])

# mostre o balanceamento das classes em 'category_desc'
print(volunteer['category_desc'].value_counts(), '\n', '\n')

# Crie um DataFrame com todas as colunas, com exceção de ``category_desc``
X = volunteer.drop('category_desc', axis=1)

# Crie um dataframe de labels com a coluna category_desc
y = volunteer[['category_desc']]

# # Utiliza a a amostragem stratificada para separar o dataset em treino e teste
test_size = 0.2
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size, random_seed=1
)

# mostre o balanceamento das classes em 'category_desc' novamente
print(y_train['category_desc'].value_counts())