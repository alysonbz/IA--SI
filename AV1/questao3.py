import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import cdist
from collections import Counter

# Carregar o DataFrame
df = pd.read_csv(r"C:\Users\vitor\Downloads\IA.BLACK\IA--SI\AV1\star_classification.csv")

# Definir as colunas relevantes
colunas_relevantes = ['alpha', 'delta', 'u', 'g', 'r', 'i', 'z', 'redshift', 'class']
df_final = df[colunas_relevantes]

# Dividir o dataset em treino e teste
X = df_final.drop(columns=['class'])
y = df_final['class']

# Corrigir valores negativos e ausentes antes da normalização
X_fixed = X.copy()

# Substituir valores negativos por zero
X_fixed = X_fixed.clip(lower=0)

# Preencher valores faltantes com a mediana
X_fixed = X_fixed.fillna(X_fixed.median(numeric_only=True))

# Dividir os dados ajustados em treino e teste
X_train_fixed, X_test_fixed, y_train, y_test = train_test_split(X_fixed, y, test_size=0.3, random_state=42)

# Função para KNN otimizada

def knn_optimized(X_train, y_train, X_test, k, distance_metric, VI=None, block_size=1000):
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
    distance_functions = {
        "mahalanobis": lambda x, y: cdist(x, y, metric="mahalanobis", VI=VI),
        "chebyshev": lambda x, y: cdist(x, y, metric="chebyshev"),
        "manhattan": lambda x, y: cdist(x, y, metric="cityblock"),
        "euclidean": lambda x, y: cdist(x, y, metric="euclidean")
    }

    if distance_metric not in distance_functions:
        raise ValueError(f"Métrica de distância '{distance_metric}' não suportada.")

    y_pred = []

    for start in range(0, len(X_test), block_size):
        end = min(start + block_size, len(X_test))
        X_test_block = X_test[start:end]

        # Calcular distâncias para o bloco atual
        distances = distance_functions[distance_metric](X_test_block, X_train)

        # Obter os índices dos k vizinhos mais próximos
        k_nearest_indices = np.argsort(distances, axis=1)[:, :k]

        # Coletar os labels correspondentes e prever os mais comuns
        for indices in k_nearest_indices:
            k_nearest_labels = y_train.iloc[indices]
            y_pred.append(Counter(k_nearest_labels).most_common(1)[0][0])

    return y_pred

# Melhor métrica de distância identificada
best_metric = "euclidean"

# Normalização Logarítmica
X_log = np.log1p(X_fixed)
X_train_log, X_test_log, _, _ = train_test_split(X_log, y, test_size=0.3, random_state=42)

# Normalização de Média Zero e Variância Unitária
X_standard = (X_fixed - X_fixed.mean()) / X_fixed.std()
X_train_std, X_test_std, _, _ = train_test_split(X_standard, y, test_size=0.3, random_state=42)

# Calcular a matriz inversa de covariância (se necessário)
VI_log = np.linalg.inv(np.cov(X_train_log.values.T)) if best_metric == "mahalanobis" else None
VI_std = np.linalg.inv(np.cov(X_train_std.values.T)) if best_metric == "mahalanobis" else None

# Avaliar KNN com normalização logarítmica
y_pred_log = knn_optimized(X_train_log.values, y_train, X_test_log.values, k=7, distance_metric=best_metric, VI=VI_log)
accuracy_log = np.mean(np.array(y_pred_log) == y_test.values)

# Avaliar KNN com normalização de média zero e variância unitária
y_pred_std = knn_optimized(X_train_std.values, y_train, X_test_std.values, k=7, distance_metric=best_metric, VI=VI_std)
accuracy_std = np.mean(np.array(y_pred_std) == y_test.values)

# Comparar os resultados
print("\nResultados da Normalização:")
print(f"Acurácia com Normalização Logarítmica: {accuracy_log:.2f}")
print(f"Acurácia com Normalização de Média Zero e Variância Unitária: {accuracy_std:.2f}")
import matplotlib.pyplot as plt

# Dados
normalizacoes = ['Logarítmica', 'Média Zero e Variância Unitária']
acuracias = [accuracy_log, accuracy_std]

# Gráfico de barras
plt.bar(normalizacoes, acuracias, color=['blue', 'green'])
plt.xlabel('Técnica de Normalização')
plt.ylabel('Acurácia')
plt.title('Comparação de Acurácias com Diferentes Técnicas de Normalização')
plt.ylim(0, 1)
plt.show()