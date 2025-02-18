import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.linear_model import Lasso

data = pd.read_csv("Dataset_coletado.csv")

if "blue" in data.columns:
    y = data["blue"]  # Alvo
    X = data.drop(columns=["blue"])  # Atributos
else:
    X = data


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


lasso = Lasso(alpha=0.01)  # Alpha controla a penalização
lasso.fit(X_scaled, y)

# Pegando os dois atributos mais relevantes
feature_importance = np.abs(lasso.coef_)
top_2_indices = np.argsort(feature_importance)[-2:]  # Pegamos os 2 maiores coeficientes
selected_features = X.columns[top_2_indices]

print(f"Atributos mais importantes selecionados pelo Lasso: {selected_features.tolist()}")

# Criando um novo conjunto de dados apenas com os dois atributos selecionados
X_selected = X_scaled[:, top_2_indices]

inertias = []
k_range = range(2, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_selected)
    inertias.append(kmeans.inertia_)

# Plotando o Método do Cotovelo
plt.figure(figsize=(8, 4))
plt.plot(k_range, inertias, 'bo-', linestyle="--", label="Inércia")
plt.xlabel("Número de Clusters (k)")
plt.ylabel("Inércia")
plt.title("Método do Cotovelo após Seleção de Atributos")
plt.legend()
plt.grid()
plt.show()

silhouette_scores = []

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_selected)
    silhouette_scores.append(silhouette_score(X_selected, labels))

# Plotando o Coeficiente de Silhueta
plt.figure(figsize=(8, 4))
plt.plot(k_range, silhouette_scores, 'ro-', linestyle="--", label="Silhueta")
plt.xlabel("Número de Clusters (k)")
plt.ylabel("Coeficiente de Silhueta")
plt.title("Análise do Coeficiente de Silhueta após Seleção de Atributos")
plt.legend()
plt.grid()
plt.show()

best_k = k_range[np.argmax(silhouette_scores)]  # Melhor k pela silhueta
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
labels = kmeans.fit_predict(X_selected)

plt.figure(figsize=(8, 6))
plt.scatter(X_selected[:, 0], X_selected[:, 1], c=labels, cmap="viridis", edgecolors="k", alpha=0.75)
plt.xlabel(selected_features[0])
plt.ylabel(selected_features[1])
plt.title(f"Clusters com k={best_k} usando os atributos selecionados")
plt.colorbar(label="Cluster")
plt.show()