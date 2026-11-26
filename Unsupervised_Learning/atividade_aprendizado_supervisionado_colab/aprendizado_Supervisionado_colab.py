# IMPORTAÇÃO DE BIBLIOTECAS

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris, load_wine, make_blobs, make_moons
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import silhouette_score, adjusted_rand_score

# 3.1 Iris
iris = load_iris()
data_iris = pd.DataFrame(iris.data, columns=iris.feature_names)
data_iris.head(10)

# 3.2 Dataset Sintético (make_blobs) - 8 centros
X_sintetico, y_sintetico = make_blobs(
    n_samples=300, centers=8, cluster_std=0.60, random_state=0, n_features=2
)
data_sintetico = pd.DataFrame(X_sintetico, columns=['Feature 1', 'Feature 2'])
data_sintetico.head()

# 3.3 Wine
wine = load_wine()
data_wine = pd.DataFrame(wine.data, columns=wine.feature_names)
data_wine.head()

kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_sintetico, y_sintetico)
data_sintetico['Cluster'] = kmeans.fit_predict(X_sintetico)

centroids = kmeans.cluster_centers_

plt.figure(figsize=(8, 6))
plt.scatter(data_sintetico['Feature 1'], data_sintetico['Feature 2'],
            c=data_sintetico['Cluster'], cmap='viridis', s=50, alpha=0.6)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=200, marker='X', label='Centróides')
plt.title('Clusterização com KMeans no Dataset Sintético')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()

inertias = []
K_range = range(1, 11)

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_sintetico)
    inertias.append(km.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(K_range, inertias, 'bo-', markersize=8)
plt.xlabel('Número de clusters (k)')
plt.ylabel('Inércia (Soma das distâncias ao quadrado)')
plt.title('Método do Cotovelo - Dataset Sintético')
plt.xticks(K_range)
plt.grid(True)
plt.show()


silhouette_scores = []

for k in range(2, 11):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_sintetico)
    score = silhouette_score(X_sintetico, labels)
    silhouette_scores.append(score)
    print(f'k={k} -> Silhouette Score: {score:.4f}')

plt.figure(figsize=(8, 5))
plt.plot(range(2, 11), silhouette_scores, 'ro-', markersize=8)
plt.xlabel('Número de clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score por número de clusters - Dataset Sintético')
plt.xticks(range(2, 11))
plt.grid(True)
plt.show()

best_k = range(2, 11)[np.argmax(silhouette_scores)]
print(f'\nMelhor k segundo Silhouette: {best_k}')

X_iris = iris.data
scaler_iris = StandardScaler()
X_iris_scaled = scaler_iris.fit_transform(X_iris)

kmeans_iris = KMeans(n_clusters=3, random_state=42, n_init=10)
labels_iris = kmeans_iris.fit_predict(X_iris_scaled)

print('Silhouette Score (Iris, k=3):', silhouette_score(X_iris_scaled, labels_iris))
print('ARI (Iris vs rótulos reais):', adjusted_rand_score(iris.target, labels_iris))

# Visualização 2D (usando as duas primeiras features)
plt.figure(figsize=(8, 6))
plt.scatter(X_iris_scaled[:, 0], X_iris_scaled[:, 1],
            c=labels_iris, cmap='viridis', s=60, alpha=0.7)
plt.title('KMeans no Iris (features 0 e 1)')
plt.xlabel('Sepal length (padronizado)')
plt.ylabel('Sepal width (padronizado)')
plt.show()


# 5.1 DBSCAN no dataset sintético
dbscan = DBSCAN(eps=0.8, min_samples=5)
labels_dbscan = dbscan.fit_predict(X_sintetico)

n_clusters_db = len(set(labels_dbscan)) - (1 if -1 in labels_dbscan else 0)
n_noise_db = list(labels_dbscan).count(-1)
print(f'DBSCAN - clusters encontrados: {n_clusters_db}')
print(f'DBSCAN - pontos de ruído: {n_noise_db}')

plt.figure(figsize=(8, 6))
plt.scatter(data_sintetico['Feature 1'], data_sintetico['Feature 2'],
            c=labels_dbscan, cmap='plasma', s=50, alpha=0.7)
plt.title('DBSCAN no Dataset Sintético (eps=0.8, min_samples=5)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

if n_clusters_db > 1:
    mask = labels_dbscan != -1
    print('Silhouette (DBSCAN, sem ruído):',
          silhouette_score(X_sintetico[mask], labels_dbscan[mask]))


# 5.2 DBSCAN em dados em forma de "luas" (make_moons)
X_moons, y_moons = make_moons(n_samples=300, noise=0.05, random_state=42)

dbscan_moons = DBSCAN(eps=0.2, min_samples=5)
labels_moons = dbscan_moons.fit_predict(X_moons)

plt.figure(figsize=(8, 6))
plt.scatter(X_moons[:, 0], X_moons[:, 1], c=labels_moons, cmap='plasma', s=50, alpha=0.7)
plt.title('DBSCAN no dataset "Moons"')
plt.xlabel('X1')
plt.ylabel('X2')
plt.show()

# Comparação: KMeans falha em dados não convexos
kmeans_moons = KMeans(n_clusters=2, random_state=42, n_init=10)
labels_km_moons = kmeans_moons.fit_predict(X_moons)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].scatter(X_moons[:, 0], X_moons[:, 1], c=labels_km_moons, cmap='viridis', s=50, alpha=0.7)
axes[0].set_title('KMeans em Moons (falha)')
axes[1].scatter(X_moons[:, 0], X_moons[:, 1], c=labels_moons, cmap='plasma', s=50, alpha=0.7)
axes[1].set_title('DBSCAN em Moons (acerta)')
plt.show()



# 6.1 Dendrograma no dataset sintético
Z = linkage(X_sintetico, method='ward')

plt.figure(figsize=(12, 6))
dendrogram(Z, truncate_mode='lastp', p=30)
plt.title('Dendrograma - Dataset Sintético (Ward)')
plt.xlabel('Amostras (ou agrupamentos)')
plt.ylabel('Distância')
plt.axhline(y=10, color='red', linestyle='--', label='Corte sugerido')
plt.legend()
plt.show()


# 6.2 AgglomerativeClustering no dataset sintético
agg = AgglomerativeClustering(n_clusters=3, linkage='ward')
labels_agg = agg.fit_predict(X_sintetico)

plt.figure(figsize=(8, 6))
plt.scatter(data_sintetico['Feature 1'], data_sintetico['Feature 2'],
            c=labels_agg, cmap='viridis', s=50, alpha=0.7)
plt.title('Clusterização Hierárquica (Ward, k=3) - Dataset Sintético')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

print('Silhouette (Hierárquica, sintético):',
      silhouette_score(X_sintetico, labels_agg))


# 6.3 Hierárquica no Iris
agg_iris = AgglomerativeClustering(n_clusters=3, linkage='ward')
labels_agg_iris = agg_iris.fit_predict(X_iris_scaled)

print('Silhouette (Hierárquica, Iris):', silhouette_score(X_iris_scaled, labels_agg_iris))
print('ARI (Hierárquica, Iris):', adjusted_rand_score(iris.target, labels_agg_iris))

plt.figure(figsize=(8, 6))
plt.scatter(X_iris_scaled[:, 0], X_iris_scaled[:, 1],
            c=labels_agg_iris, cmap='viridis', s=60, alpha=0.7)
plt.title('Hierárquica no Iris (features 0 e 1)')
plt.show()


resultados = []

# Sintético
resultados.append({
    'Dataset': 'Sintético',
    'Algoritmo': 'KMeans',
    'Silhouette': silhouette_score(X_sintetico, KMeans(n_clusters=3, random_state=42, n_init=10).fit_predict(X_sintetico))
})
resultados.append({
    'Dataset': 'Sintético',
    'Algoritmo': 'Hierárquica',
    'Silhouette': silhouette_score(X_sintetico, labels_agg)
})
# DBSCAN (se encontrou >1 cluster)
if n_clusters_db > 1:
    mask = labels_dbscan != -1
    resultados.append({
        'Dataset': 'Sintético',
        'Algoritmo': 'DBSCAN',
        'Silhouette': silhouette_score(X_sintetico[mask], labels_dbscan[mask])
    })

# Iris
resultados.append({
    'Dataset': 'Iris',
    'Algoritmo': 'KMeans',
    'Silhouette': silhouette_score(X_iris_scaled, labels_iris)
})
resultados.append({
    'Dataset': 'Iris',
    'Algoritmo': 'Hierárquica',
    'Silhouette': silhouette_score(X_iris_scaled, labels_agg_iris)
})

df_resultados = pd.DataFrame(resultados)
print(df_resultados)

plt.figure(figsize=(10, 5))
sns.barplot(data=df_resultados, x='Dataset', y='Silhouette', hue='Algoritmo')
plt.title('Comparação de Silhouette Score entre algoritmos')
plt.ylim(0, 1)
plt.legend(title='Algoritmo')
plt.show()
