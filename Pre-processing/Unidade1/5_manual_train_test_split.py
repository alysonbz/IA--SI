from unicodedata import category
import pandas as pd
from sklearn.model_selection import train_test_split as sklearn_split

from src.utils import load_volunteer_dataset


def train_test_split(X, y, test_size=0.2, stratify=None, random_state=None):
    # Implementação interna da função usando o scikit-learn
    return sklearn_split(X, y, test_size=test_size, stratify=stratify, random_state=random_state)


volunteer = load_volunteer_dataset()

# 1. Exclua as colunas Latitude e Longitude de volunteer
volunteer_new = volunteer.drop(columns=['Latitude', 'Longitude'])

# 2. Exclua as linhas com valores null da coluna category_desc de volunteer_new
volunteer = volunteer_new.dropna(subset=['category_desc'])

# 3. Mostre o balanceamento das classes em 'category_desc'
print(volunteer['category_desc'].value_counts(), '\n', '\n')

# 4. Crie um DataFrame com todas as colunas, com exceção de `category_desc`
X = volunteer.drop('category_desc', axis=1)

# 5. Crie um dataframe de labels com a coluna category_desc
y = volunteer[['category_desc']]

# 6. Utiliza a amostragem estratificada para separar o dataset em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

# 7. Mostre o balanceamento das classes em 'category_desc' novamente (no conjunto de treino)
print(y_train['category_desc'].value_counts())
