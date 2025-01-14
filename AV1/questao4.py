import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.model_selection import train_test_split
from collections import Counter
from joblib import Parallel, delayed

# Carregar o DataFrame
df = pd.read_csv(r"C:\Users\vitor\Downloads\IA.BLACK\IA--SI\AV1\star_classification.csv")

# Definir as colunas relevantes
colunas_relevantes = ['alpha', 'delta', 'u', 'g', 'r', 'i', 'z', 'redshift', 'class']
df_final = df[colunas_relevantes]

# Preprocessamento: corrigir valores inválidos
X = df_final.drop(columns=['class'])
X = X.apply(lambda x: np.where(x < 0, 0, x) if np.issubdtype(x.dtype, np.number) else x)  # Substituir valores negativos por 0
X = X.fillna(X.median(numeric_only=True))  # Preencher valores ausentes com a mediana

y = df_final['class']

# Normalização escolhida (logarítmica)
X_normalized = X.apply(lambda x: np.log1p(x) if np.issubdtype(x.dtype, np.number) else x)

# Divisão treino-teste
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.3, random_state=42)

# Função KNN com cálculo paralelizado e otimizado
def knn_parallel_optimized(X_train, y_train, X_test, k, distance_metric, block_size=500, VI=None, n_jobs=-1):
    distance_metrics = {
        "mahalanobis": lambda x, y: cdist(x, y, metric="mahalanobis", VI=VI),
        "chebyshev": lambda x, y: cdist(x, y, metric="chebyshev"),
        "manhattan": lambda x, y: cdist(x, y, metric="cityblock"),
        "euclidean": lambda x, y: cdist(x, y, metric="euclidean"),
    }

    if distance_metric not in distance_metrics:
        raise ValueError(f"Métrica de distância '{distance_metric}' não suportada.")

    def process_block(start, end):
        X_test_block = X_test[start:end]
        distances = distance_metrics[distance_metric](X_test_block, X_train)
        k_nearest_indices = np.argsort(distances, axis=1)[:, :k]
        return [
            Counter(y_train.iloc[indices]).most_common(1)[0][0]
            for indices in k_nearest_indices
        ]

    # Dividir os índices do conjunto de teste em blocos menores
    num_blocks = len(X_test) // block_size + (len(X_test) % block_size > 0)
    blocks = [(i * block_size, min((i + 1) * block_size, len(X_test))) for i in range(num_blocks)]

    # Processar blocos em paralelo com limitação de recursos
    results = Parallel(n_jobs=n_jobs, prefer="threads")(delayed(process_block)(start, end) for start, end in blocks)

    # Combinar resultados
    return [pred for block in results for pred in block]

# Melhor métrica de distância identificada
best_metric = "euclidean"

# Avaliar a acurácia para diferentes valores de k
k_values = range(1, 21)
accuracies = []

for k in k_values:
    y_pred = knn_parallel_optimized(X_train.values, y_train, X_test.values, k=k, distance_metric=best_metric, block_size=500, n_jobs=4)
    accuracy = np.mean(np.array(y_pred) == y_test.values)
    accuracies.append(accuracy)

# Encontrar o melhor k
best_k = k_values[np.argmax(accuracies)]
best_accuracy = max(accuracies)

# Plotar o gráfico
plt.figure(figsize=(10, 6))
plt.plot(k_values, accuracies, marker='o', linestyle='-', color='b')
plt.title("Acurácia do KNN para diferentes valores de k")
plt.xlabel("Número de vizinhos (k)")
plt.ylabel("Acurácia")
plt.xticks(k_values)
plt.grid()
plt.axvline(best_k, color='r', linestyle='--', label=f'Melhor k = {best_k}')
plt.legend()
plt.show()

# Exibir o melhor k e a acurácia correspondente
print(f"Melhor k: {best_k}")
print(f"Acurácia com o melhor k: {best_accuracy:.2f}")