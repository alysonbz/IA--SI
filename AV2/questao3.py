import numpy as np
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from AV2.src.utills import diabetes_dataset

diabetes = diabetes_dataset()

X = diabetes.drop("Outcome", axis=1)
y = diabetes["Outcome"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

lasso = Lasso(alpha=0.1, random_state=42)
lasso.fit(X_scaled, y)

coef = np.abs(lasso.coef_)
most_relevant_indices = np.argsort(coef)[-2:]  # Obter os dois maiores coeficientes
most_relevant_features = X.columns[most_relevant_indices]
print("Atributos mais relevantes com Lasso:", list(most_relevant_features))

X_reduced = X_scaled[:, most_relevant_indices]

inertia = []
silhouette_scores = []
range_n_clusters = range(2, 11)

for k in range_n_clusters:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_reduced)

    inertia.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_reduced, kmeans.labels_))

plt.figure(figsize=(12, 6))
plt.plot(range_n_clusters, inertia, marker='o', linestyle='--')
plt.title("Método do Cotovelo com Atributos Selecionados (Lasso)")
plt.xlabel("Número de Clusters (k)")
plt.ylabel("Inertia")
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(range_n_clusters, silhouette_scores, marker='o', linestyle='--', color='orange')
plt.title("Método da Silhueta com Atributos Selecionados (Lasso)")
plt.xlabel("Número de Clusters (k)")
plt.ylabel("Coeficiente de Silhueta")
plt.show()

best_k_silhouette = range_n_clusters[silhouette_scores.index(max(silhouette_scores))]
print("Melhor k com base no Método da Silhueta:", best_k_silhouette)

kmeans_silhouette = KMeans(n_clusters=best_k_silhouette, random_state=42).fit(X_reduced)
plt.figure(figsize=(8, 6))
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=kmeans_silhouette.labels_, cmap="viridis", s=50)
plt.title(f"Scatterplot com k={best_k_silhouette} (Método da Silhueta)")
plt.xlabel(most_relevant_features[0])
plt.ylabel(most_relevant_features[1])
plt.show()

best_k_elbow = 4
kmeans_elbow = KMeans(n_clusters=best_k_elbow, random_state=42).fit(X_reduced)
plt.figure(figsize=(8, 6))
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=kmeans_elbow.labels_, cmap="viridis", s=50)
plt.title(f"Scatterplot com k={best_k_elbow} (Método do Cotovelo)")
plt.xlabel(most_relevant_features[0])
plt.ylabel(most_relevant_features[1])
plt.show()