from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from AV2.src.utills import diabetes_dataset

diabetes = diabetes_dataset()

X = diabetes.drop("Outcome", axis=1)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

inertia = []
silhouette_scores = []

range_n_clusters = range(2, 11)  # Testar de 2 a 10 clusters
for k in range_n_clusters:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)

    inertia.append(kmeans.inertia_)

    silhouette_avg = silhouette_score(X_scaled, kmeans.labels_)
    silhouette_scores.append(silhouette_avg)

plt.figure(figsize=(12, 6))
plt.plot(range_n_clusters, inertia, marker='o', linestyle='--')
plt.title("Método do Cotovelo: Soma das Distâncias Quadradas")
plt.xlabel("Número de Clusters (k)")
plt.ylabel("Inertia")
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(range_n_clusters, silhouette_scores, marker='o', linestyle='--', color='orange')
plt.title("Coeficiente de Silhueta para Diferentes k")
plt.xlabel("Número de Clusters (k)")
plt.ylabel("Coeficiente de Silhueta")
plt.show()

best_k = range_n_clusters[silhouette_scores.index(max(silhouette_scores))]
print("Melhor k com base no Método da Silhueta:", best_k)