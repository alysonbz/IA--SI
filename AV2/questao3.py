import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Carregar o dataset
classificacao = pd.read_csv('./dataset/dataset_classificacao_ajustado.csv')

# Separar variáveis independentes (X) e dependente (y)
X = classificacao.drop(columns=["booking_status"])
y = classificacao["booking_status"]

# Padronizar os dados
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Aplicar Lasso para seleção de atributos
lasso = Lasso(alpha=0.01)
lasso.fit(X_scaled, y)

# Obter os coeficientes e selecionar os dois atributos mais relevantes
coeficientes = pd.Series(lasso.coef_, index=X.columns)
top_2_features = coeficientes.abs().nlargest(2).index.tolist()

print("Os dois atributos mais relevantes são:", top_2_features)

# Selecionar os dois atributos no dataset
X_selected = classificacao[top_2_features]

# # Método do Cotovelo
inertia = []

for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_selected)
    inertia.append(kmeans.inertia_)

# Determinar o melhor k com o método do cotovelo
best_k_elbow = 3

# Plot do gráfico do método do cotovelo
plt.figure(figsize=(8, 6))
plt.plot(range(1, 11), inertia, marker='o', linestyle='-')
plt.xlabel('Número de Clusters (K)')
plt.ylabel('Inércia (Soma das Distâncias ao Centroide)')
plt.title('Método do Cotovelo')
plt.grid(True)
plt.show()

# Aplicar KMeans com o melhor k (Cotovelo)
kmeans_elbow = KMeans(n_clusters=best_k_elbow, random_state=42)
classificacao['Cluster_Elbow'] = kmeans_elbow.fit_predict(X_selected)

# Obter os centróides dos clusters
centroids = kmeans_elbow.cluster_centers_

# Scatterplot com os clusters do Cotovelo
plt.figure(figsize=(8, 6))
plt.scatter(X_selected.iloc[:, 0], X_selected.iloc[:, 1], c=classificacao['Cluster_Elbow'], cmap='viridis', s=50, alpha=0.6, label='Pontos')
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=200, label='Centróides')  # Plotando os centróides

plt.title(f'Clusters pelo Método do Cotovelo (K={best_k_elbow})')
plt.xlabel(top_2_features[0])
plt.ylabel(top_2_features[1])
plt.legend()
plt.show()

# OBS:
# Os metódos são rodados separadamente, primeiro o Cotovelo e só depois o Shilhouette, para não haver sobrercarga no computador.
# Por isso é comentado o metódo que não vai ser usado na análise.

# # Método da Silhueta
# silhouette_scores = []
#
# # Calcular o Silhouette Score para diferentes valores de k
# for k in range(2, 11):
#     kmeans = KMeans(n_clusters=k, random_state=42)
#     labels = kmeans.fit_predict(X_selected)
#     silhouette_scores.append(silhouette_score(X_selected, labels))
#
# # Determinar o melhor k com o método de Silhueta
# best_k_silhouette = silhouette_scores.index(max(silhouette_scores)) + 2
#
# # Plot do gráfico do coeficiente de silhueta
# plt.figure(figsize=(8, 6))
# plt.plot(range(2, 11), silhouette_scores, marker='o', linestyle='-')
# plt.xlabel('Número de Clusters (K)')
# plt.ylabel('Coeficiente de Silhueta')
# plt.title('Análise pelo Coeficiente de Silhueta')
# plt.grid(True)
# plt.show()
#
# # Aplicar KMeans com o melhor k (Silhueta)
# kmeans_silhouette = KMeans(n_clusters=best_k_silhouette, random_state=42)
# classificacao['Cluster_Silhouette'] = kmeans_silhouette.fit_predict(X_selected)
#
# # Obter os centróides dos clusters
# centroids = kmeans_silhouette.cluster_centers_
#
# # Scatterplot com os clusters da Silhueta
# plt.figure(figsize=(8, 6))
# plt.scatter(X_selected.iloc[:, 0], X_selected.iloc[:, 1], c=classificacao['Cluster_Silhouette'], cmap='viridis', s=50, alpha=0.6)
#
# # Plotando os centróides
# plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=200, label='Centróides')
#
# plt.title(f'Clusters pelo Método da Silhueta (K={best_k_silhouette})')
# plt.xlabel(top_2_features[0])
# plt.ylabel(top_2_features[1])
# plt.legend()
# plt.show()