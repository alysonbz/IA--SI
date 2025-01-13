import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import mahalanobis, chebyshev, cityblock, euclidean
from collections import Counter

# 1. Importe as bibliotecas necessárias
# (importações acima já cobrem esta etapa)

# 2. Carregar o dataset atualizado ou original
try:
    # Tentar carregar o dataset atualizado
    df = pd.read_csv("gender_classification_adjusted.csv")
    print("Dataset atualizado carregado.")
except FileNotFoundError:
    # Carregar o dataset original caso o atualizado não esteja disponível
    url = "https://www.kaggleusercontent.com/datasets/elakiricoder/gender-classification-dataset/data.csv"
    df = pd.read_csv(url)
    print("Dataset original carregado.")

# Manter apenas as colunas relevantes
columns_needed = ['height', 'weight', 'gender']
df = df[columns_needed]

# Converter classes para valores numéricos, se necessário
df['gender'] = df['gender'].map({'Male': 0, 'Female': 1})

# 3. Dividir o dataset em treino e teste sem normalizar
X = df[['height', 'weight']].values
y = df['gender'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# 4. Implementar o KNN manualmente
def knn_predict(X_train, y_train, X_test, k, metric):
    """
    Implementa o algoritmo KNN de forma manual.
    """
    y_pred = []

    # Calcular matriz de covariância para distância de Mahalanobis
    cov_matrix = np.cov(X_train, rowvar=False) if metric == 'mahalanobis' else None
    inv_cov_matrix = np.linalg.inv(cov_matrix) if cov_matrix is not None else None

    for test_point in X_test:
        distances = []
        for i, train_point in enumerate(X_train):
            if metric == 'mahalanobis':
                dist = mahalanobis(test_point, train_point, inv_cov_matrix)
            elif metric == 'chebyshev':
                dist = chebyshev(test_point, train_point)
            elif metric == 'manhattan':
                dist = cityblock(test_point, train_point)
            elif metric == 'euclidean':
                dist = euclidean(test_point, train_point)
            else:
                raise ValueError("Métrica de distância inválida.")

            distances.append((dist, y_train[i]))

        # Ordenar as distâncias e pegar os k vizinhos mais próximos
        distances.sort(key=lambda x: x[0])
        k_neighbors = [neighbor[1] for neighbor in distances[:k]]

        # Determinar a classe majoritária entre os k vizinhos
        most_common = Counter(k_neighbors).most_common(1)[0][0]
        y_pred.append(most_common)

    return np.array(y_pred)


# Função para calcular a acurácia
def calculate_accuracy(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)


# Comparar acurácias para diferentes métricas
k = 7
metrics = ['mahalanobis', 'chebyshev', 'manhattan', 'euclidean']
accuracies = {}

for metric in metrics:
    print(f"\nCalculando com métrica: {metric}")
    y_pred = knn_predict(X_train, y_train, X_test, k, metric)
    accuracy = calculate_accuracy(y_test, y_pred)
    accuracies[metric] = accuracy
    print(f"Acurácia para {metric}: {accuracy:.4f}")

# Exibir resultados
print("\nResumo das acurácias:")
for metric, acc in accuracies.items():
    print(f"{metric.capitalize()}: {acc:.4f}")