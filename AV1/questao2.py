import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from scipy.spatial.distance import cdist
from collections import Counter

# Carregar os dados
df = pd.read_csv(r"C:\Users\vitor\Downloads\IA.BLACK\IA--SI\AV1\star_classification.csv")

# Seleção das colunas relevantes
colunas_relevantes = ['alpha', 'delta', 'u', 'g', 'r', 'i', 'z', 'redshift', 'class']
df_final = df[colunas_relevantes]

X = df_final.drop(columns=['class'])
y = df_final['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

def knn_blockwise(X_train, y_train, X_test, k, distance_metric, VI=None, block_size=1000):
    """
    Implementação otimizada do k-NN com cálculo em blocos para economizar memória.

    Parâmetros:
        X_train (ndarray): Dados de treinamento.
        y_train (Series): Labels de treinamento.
        X_test (ndarray): Dados de teste.
        k (int): Número de vizinhos.
        distance_metric (str): Métrica de distância.
        VI (ndarray, opcional): Matriz de inversa para distância de Mahalanobis.
        block_size (int): Tamanho do bloco para cálculo de distâncias.

    Retorna:
        y_pred (list): Previsões para os dados de teste.
    """
    distance_metrics = {
        "mahalanobis": lambda x, y: cdist(x, y, metric="mahalanobis", VI=VI),
        "chebyshev": lambda x, y: cdist(x, y, metric="chebyshev"),
        "manhattan": lambda x, y: cdist(x, y, metric="cityblock"),
        "euclidean": lambda x, y: cdist(x, y, metric="euclidean")
    }

    if distance_metric not in distance_metrics:
        raise ValueError(f"Métrica de distância '{distance_metric}' não suportada.")

    y_pred = []

    for start in range(0, len(X_test), block_size):
        end = min(start + block_size, len(X_test))
        X_test_block = X_test[start:end]

        # Calcular distâncias para o bloco atual
        distances = distance_metrics[distance_metric](X_test_block, X_train)

        # Obter os índices dos k vizinhos mais próximos
        k_nearest_indices = np.argsort(distances, axis=1)[:, :k]

        # Coletar os labels correspondentes e prever os mais comuns
        for indices in k_nearest_indices:
            k_nearest_labels = y_train.iloc[indices]
            y_pred.append(Counter(k_nearest_labels).most_common(1)[0][0])

    return y_pred

# Calcular a matriz de inversa para a distância de Mahalanobis
VI = np.linalg.inv(np.cov(X_train.values.T))

results = []
for metric in ["mahalanobis", "chebyshev", "manhattan", "euclidean"]:
    print(f"\nUsando {metric.capitalize()}")
    y_pred = knn_blockwise(
        X_train.values, y_train, X_test.values, k=7,
        distance_metric=metric, VI=VI if metric == "mahalanobis" else None,
        block_size=1000
    )
    accuracy = np.mean(np.array(y_pred) == y_test.values)  # Acurácia
    results.append((metric, accuracy))
    print(f"Acurácia: {accuracy:.2f}")

print("\nResumo dos resultados:")
for metric, acc in results:
    print(f"Métrica: {metric.capitalize()}, Acurácia: {acc:.2f}")
import matplotlib.pyplot as plt

metrics = ["Mahalanobis", "Chebyshev", "Manhattan", "Euclidean"]
accuracies = [result[1] for result in results]

plt.figure(figsize=(10, 6))
plt.bar(metrics, accuracies, color=["blue", "orange", "green", "red"])
plt.title("Comparação de Acurácias por Métrica de Distância")
plt.xlabel("Métricas de Distância")
plt.ylabel("Acurácia")
plt.ylim(0.8, 0.9)
plt.grid(axis='y')
plt.show()