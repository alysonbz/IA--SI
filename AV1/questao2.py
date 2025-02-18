import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.spatial.distance import mahalanobis, chebyshev, cityblock, euclidean

# Funções de distância
def mahalanobis_distance(x, y, inv_cov_matrix):
    return mahalanobis(x, y, inv_cov_matrix)

def chebyshev_distance(x, y):
    return chebyshev(x, y)

def manhattan_distance(x, y):
    return cityblock(x, y)

def euclidean_distance(x, y):
    return euclidean(x, y)

#KNN
def knn_manual_optimized(X_train, y_train, X_test, k, distance_func, inv_cov_matrix=None):
    y_pred = []
    # Limitar o tamanho do conjunto de treino para otimizar o cálculo
    sample_size = min(len(X_train), 5000)  # Usar no máximo 5000 amostras
    X_train_sample = X_train[:sample_size]
    y_train_sample = y_train[:sample_size]

    for test_point in X_test:
        distances = []
        for i, train_point in enumerate(X_train_sample):
            if distance_func == mahalanobis_distance:
                dist = distance_func(test_point, train_point, inv_cov_matrix)
            else:
                dist = distance_func(test_point, train_point)
            distances.append((dist, y_train_sample[i]))
        distances = sorted(distances, key=lambda x: x[0])[:k]
        classes = [neighbor[1] for neighbor in distances]
        y_pred.append(max(set(classes), key=classes.count))
    return np.array(y_pred)


# Carregar o dataset atualizado
classificacao = pd.read_csv('./dataset/dataset_classificacao_ajustado.csv')

# Dividir em X (entradas) e y (saídas)
X = classificacao.drop(columns=['booking_status'], axis=1).values
y = classificacao['booking_status'].values

# Divisão treino/teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=y, random_state=42)

# Calcular a matriz inversa da covariância para Mahalanobis
cov_matrix = np.cov(X_train, rowvar=False)
inv_cov_matrix = np.linalg.inv(cov_matrix)

# Valor de k
k = 7

# Distâncias para teste
distances = {
    "Mahalanobis": lambda x, y: mahalanobis_distance(x, y, inv_cov_matrix),
    "Chebyshev": chebyshev_distance,
    "Manhattan": manhattan_distance,
    "Euclidiana": euclidean_distance
}

# Avaliar e comparar
results = {}
for name, func in distances.items():
    print(f"Calculando com distância: {name}...")
    y_pred = knn_manual_optimized(X_train, y_train, X_test, k, func, inv_cov_matrix)
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc

# Mostrar resultados
print("\nComparação de Acurácias:")
for name, acc in results.items():
    print(f"{name}: {acc:.4f}")