import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# 1️⃣ Carregar o dataset
data = pd.read_csv("Dataset_coletado.csv")

# 2️⃣ Remover a coluna alvo se existir
if "blue" in data.columns:
    data = data.drop(columns=["blue"])

# 3️⃣ Normalizar os dados
scaler = StandardScaler()
X_scaled = scaler.fit_transform(data)

# 4️⃣ Método do Cotovelo
inertias = []
k_range = range(2, 11)  # Testando valores de k de 2 a 10

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)

# Plotando o Método do Cotovelo
plt.figure(figsize=(8, 4))
plt.plot(k_range, inertias, 'bo-', linestyle="--", label="Inércia")
plt.xlabel("Número de Clusters (k)")
plt.ylabel("Inércia")
plt.title("Método do Cotovelo para Determinar k Ótimo")
plt.legend()
plt.grid()
plt.show()

# 5️⃣ Método da Silhueta
silhouette_scores = []

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    silhouette_scores.append(silhouette_score(X_scaled, labels))

# Plotando o Coeficiente de Silhueta
plt.figure(figsize=(8, 4))
plt.plot(k_range, silhouette_scores, 'ro-', linestyle="--", label="Silhueta")
plt.xlabel("Número de Clusters (k)")
plt.ylabel("Coeficiente de Silhueta")
plt.title("Análise do Coeficiente de Silhueta")
plt.legend()
plt.grid()
plt.show()