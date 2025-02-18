import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler

# Carregar o dataset
df = pd.read_csv("sabores_de_cacau_ajustado.csv")

# Remover a coluna alvo 'rating' do dataset para análise de clustering
if 'rating' in df.columns:
    X_clustering = df.drop(columns=['rating'])
else:
    X_clustering = df.copy()

# Normalizar os dados
scaler = MinMaxScaler()
X_clustering = pd.DataFrame(scaler.fit_transform(X_clustering), columns=X_clustering.columns)

# Lista de valores de k para testar
k_values = list(range(2, 11))

# Armazenar métricas
inertia = []
silhouette_scores = []

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_clustering)

    # Calcular a inércia (método do cotovelo)
    inertia.append(kmeans.inertia_)

    # Calcular a pontuação da silhueta
    silhouette_avg = silhouette_score(X_clustering, cluster_labels)
    silhouette_scores.append(silhouette_avg)

# Plotar o gráfico do método do cotovelo
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(k_values, inertia, marker='o', linestyle='--')
plt.xlabel('Número de Clusters (k)')
plt.ylabel('Inércia')
plt.title('Método do Cotovelo')

# Plotar o gráfico da pontuação da silhueta
plt.subplot(1, 2, 2)
plt.plot(k_values, silhouette_scores, marker='o', linestyle='--', color='red')
plt.xlabel('Número de Clusters (k)')
plt.ylabel('Pontuação da Silhueta')
plt.title('Média da Silhueta')

plt.show()
