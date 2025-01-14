import numpy as np
import pandas as pd
from collections import Counter

# é distância "reta" entre dois pontos no espaço
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

# é a soma das distâncias absolutas entre as coordenadas dos pontos.
def manhattan_distance(x1, x2):
    return np.sum(np.abs(x1 - x2))

# é a maior distância entre as coordenadas dos pontos.
def chebyshev_distance(x1, x2):
    return np.max(np.abs(x1 - x2))

# é uma medida de distância que leva em conta a correlação entre as variáveis.
def mahalanobis_distance(x1, x2, VI):
    delta = x1 - x2
    return np.sqrt(np.dot(np.dot(delta, VI), delta.T))

# função para encontrar os K vizinhos mais próximos
def get_neighbors(X_train, y_train, test_sample, k, distance_metric, VI=None):
    distances = []
    for i in range(len(X_train)):
        if distance_metric == 'euclidean':
            dist = euclidean_distance(test_sample, X_train[i])
        elif distance_metric == 'manhattan':
            dist = manhattan_distance(test_sample, X_train[i])
        elif distance_metric == 'chebyshev':
            dist = chebyshev_distance(test_sample, X_train[i])
        elif distance_metric == 'mahalanobis':
            dist = mahalanobis_distance(test_sample, X_train[i], VI)
        distances.append((y_train[i], dist))
    distances.sort(key=lambda x: x[1])
    neighbors = [distances[i][0] for i in range(k)]
    return neighbors

# função para realizar a classificação com base nos vizinhos mais próximos
def predict_classification(X_train, y_train, X_test, k, distance_metric, VI=None):
    predictions = []
    for test_sample in X_test:
        neighbors = get_neighbors(X_train, y_train, test_sample, k, distance_metric, VI)
        most_common = Counter(neighbors).most_common(1)[0][0]
        predictions.append(most_common)
    return predictions

def accuracy_score(y_true, y_pred):
    correct = np.sum(y_true == y_pred)
    return correct / len(y_true)

try:
    data = pd.read_csv(r"C:\Users\aryel\OneDrive\Documentos\IA_ary\IA--SI\AV1\dataset\Cancer_Data_ajustado.csv")
except FileNotFoundError:
    data = pd.read_csv(r"C:\Users\aryel\OneDrive\Documentos\IA_ary\IA--SI\AV1\dataset\Cancer_Data.csv")

X = data.drop(columns=['diagnosis']).values
y = data['diagnosis'].values

# Função manual para dividir os dados em treino e teste
def manual_train_test_split(X, y, test_size=0.2, random_state=None):
    if random_state:
        np.random.seed(random_state)
    indices = np.arange(len(X))
    np.random.shuffle(indices)

    test_set_size = int(len(X) * test_size)
    test_indices = indices[:test_set_size]
    train_indices = indices[test_set_size:]

    X_train, X_test = X[train_indices, :], X[test_indices, :]
    y_train, y_test = y[train_indices], y[test_indices]

    return X_train, X_test, y_train, y_test

# Divisão do dataset em treino e teste
X_train, X_test, y_train, y_test = manual_train_test_split(X, y, test_size=0.2, random_state=42)

# a normalização logarítmica:
if (X_train < 0).any() or (X_test < 0).any():
    raise ValueError("os dados tem val negativos, ent não da p fazer a normalização logarítmica.")

X_train_log = np.log1p(X_train)
X_test_log = np.log1p(X_test)

# Calcular a matriz de covariância inversa para a métrica de Mahalanobis
VI = np.linalg.pinv(np.cov(X_train_log.T))  # Usando pseudo-inversa para evitar erro de singularidade

# Definir o valor de k
k = 7

# Realizar a classificação com diferentes métricas de distância
y_pred_euclidean = predict_classification(X_train_log, y_train, X_test_log, k, 'euclidean')
y_pred_manhattan = predict_classification(X_train_log, y_train, X_test_log, k, 'manhattan')
y_pred_chebyshev = predict_classification(X_train_log, y_train, X_test_log, k, 'chebyshev')
y_pred_mahalanobis = predict_classification(X_train_log, y_train, X_test_log, k, 'mahalanobis', VI)

# Calcular a acurácia para cada métrica de distância
accuracy_euclidean = accuracy_score(y_test, y_pred_euclidean)
accuracy_manhattan = accuracy_score(y_test, y_pred_manhattan)
accuracy_chebyshev = accuracy_score(y_test, y_pred_chebyshev)
accuracy_mahalanobis = accuracy_score(y_test, y_pred_mahalanobis)

# Exibir as acurácias
print(f'Acurácia com distância Euclidiana: {accuracy_euclidean}')
print(f'Acurácia com distância de Manhattan: {accuracy_manhattan}')
print(f'Acurácia com distância de Chebyshev: {accuracy_chebyshev}')
print(f'Acurácia com distância de Mahalanobis: {accuracy_mahalanobis}')
