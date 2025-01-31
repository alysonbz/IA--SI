import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import mahalanobis, chebyshev, cityblock, euclidean

# 1) Importar bibliotecas

# 2) Carregar o dataset atualizado
file_path = 'waterquality_ajustado.csv'
try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"Arquivo {file_path} não encontrado.")

# Separar recursos e rótulos
X = df.drop('is_safe', axis=1).values
y = df['is_safe'].values

# 3) Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Função para calcular distâncias
DISTANCE_FUNCTIONS = {
    "mahalanobis": lambda x, y, VI: mahalanobis(x, y, VI),
    "chebyshev": lambda x, y, _: chebyshev(x, y),
    "manhattan": lambda x, y, _: cityblock(x, y),
    "euclidean": lambda x, y, _: euclidean(x, y)
}
# Calcular matriz inversa de covariância
VI = np.linalg.inv(np.cov(X_train, rowvar=False))
def knn_predict(X_train, y_train, X_test, k, distance_metric, VI=None):
    y_pred = []
    for test_point in X_test:
        distances = []
        for i, train_point in enumerate(X_train):
            dist = distance_metric(test_point, train_point, VI)
            distances.append((dist, y_train[i]))
        # Ordenar distâncias e pegar os k mais próximos
        distances.sort(key=lambda x: x[0])
        k_nearest = distances[:k]
        # Votação
        classes = [label for _, label in k_nearest]
        y_pred.append(max(set(classes), key=classes.count))
    return np.array(y_pred)

# 4) Implementar o KNN para cada métrica e exibir a acurácia
k = 7
for metric_name, distance_fn in DISTANCE_FUNCTIONS.items():
    print(f"Usando métrica de distância: {metric_name}")
    y_pred = knn_predict(X_train, y_train, X_test, k, distance_fn, VI if metric_name == "mahalanobis" else None)
    accuracy = np.mean(y_pred == y_test)
    print(f"Acurácia: {accuracy:.4f}\n")
