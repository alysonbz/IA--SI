import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# Carregar os dados (caminho correto necessário)
df = pd.read_csv(r"C:\Users\bende\av1\IA--SI\AV2\star_classification.csv")

# Usando uma amostra de 10% para acelerar o processo
df_sample = df.sample(frac=0.1, random_state=42)

# Remover a coluna alvo
X = df_sample.drop(columns=['class']).values

# Normalizar os dados
X = StandardScaler().fit_transform(X)

# Função para calcular os metrics de KMeans
def calculate_kmeans_metrics(k):
    # Usando MiniBatchKMeans com batch_size reduzido e número menor de iterações
    kmeans = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=100, max_iter=100, tol=1e-4)
    kmeans.fit(X)

    # Calculando o SSE (Soma dos Erros Quadráticos)
    sse = kmeans.inertia_

    # Calculando o Silhouette Score
    silhouette = silhouette_score(X, kmeans.labels_) if k > 1 else np.nan

    return k, sse, silhouette  # Retornar os 3 valores como uma tupla

# Testando apenas de 2 a 6 clusters para acelerar
cluster_range = range(2, 7)

# Calculando os valores de SSE e Silhueta para diferentes números de clusters
metrics = [calculate_kmeans_metrics(k) for k in cluster_range]

# Separando os resultados de SSE e Silhueta para plotagem
sse = [metric[1] for metric in metrics]
silhouette_scores = [metric[2] for metric in metrics]

# Plotando os resultados
plt.figure(figsize=(10, 5))

# Gráfico do Cotovelo
plt.subplot(1, 2, 1)
plt.plot(cluster_range, sse, marker='o')
plt.title('Método do Cotovelo')
plt.xlabel('Número de Clusters')
plt.ylabel('Soma dos Erros Quadráticos (SSE)')
plt.xticks(cluster_range)
plt.grid(True)

# Gráfico do método da Silhueta
plt.subplot(1, 2, 2)
plt.plot(cluster_range, silhouette_scores, marker='o', color='green')
plt.title('Método da Silhueta')
plt.xlabel('Número de Clusters')
plt.ylabel('Pontuação da Silhueta')
plt.xticks(cluster_range)
plt.grid(True)

plt.tight_layout()
plt.show()

# Exibir pontuação da silhueta para cada valor de k
for k, score in zip(cluster_range, silhouette_scores):
    print(f"{k} clusters: {score:.4f}")
