import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Lasso

# Carregar o dataset
df = pd.read_csv("sabores_de_cacau_ajustado.csv")

# Separar features e target
if 'rating' in df.columns:
    X = df.drop(columns=['rating'])
    y = df['rating']
else:
    X = df.copy()
    y = None

# Normalizar os dados
scaler = MinMaxScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Aplicar Lasso para selecionar os atributos mais relevantes
lasso = Lasso(alpha=0.01)
lasso.fit(X_scaled, y)

# Selecionar os dois atributos mais importantes
importance = np.abs(lasso.coef_)
top_2_features = X.columns[np.argsort(importance)[-2:]]
X_selected = X_scaled[top_2_features]

# Lista de valores de k para testar
k_values = list(range(2, 11))

# Armazenar métricas
inertia = []
silhouette_scores = []

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_selected)

    # Calcular a inércia (método do cotovelo)
    inertia.append(kmeans.inertia_)

    # Calcular a pontuação da silhueta
    silhouette_avg = silhouette_score(X_selected, cluster_labels)
    silhouette_scores.append(silhouette_avg)

# Determinar os melhores k para cada método
best_k_elbow = k_values[np.argmin(np.gradient(inertia))]
best_k_silhouette = k_values[np.argmax(silhouette_scores)]

print(f"Melhor k pelo Método do Cotovelo: {best_k_elbow}")
print(f"Melhor k pelo Método da Silhueta: {best_k_silhouette}")

# Criar scatter plots
plt.figure(figsize=(12, 5))

# Scatter plot para o método do cotovelo
kmeans_elbow = KMeans(n_clusters=best_k_elbow, random_state=42, n_init=10).fit(X_selected)
plt.subplot(1, 2, 1)
plt.scatter(X_selected.iloc[:, 0], X_selected.iloc[:, 1], c=kmeans_elbow.labels_, cmap='viridis', alpha=0.6)
plt.xlabel(top_2_features[0])
plt.ylabel(top_2_features[1])
plt.title(f'Clusters com k={best_k_elbow} (Cotovelo)')

# Scatter plot para o método da silhueta
kmeans_silhouette = KMeans(n_clusters=best_k_silhouette, random_state=42, n_init=10).fit(X_selected)
plt.subplot(1, 2, 2)
plt.scatter(X_selected.iloc[:, 0], X_selected.iloc[:, 1], c=kmeans_silhouette.labels_, cmap='viridis', alpha=0.6)
plt.xlabel(top_2_features[0])
plt.ylabel(top_2_features[1])
plt.title(f'Clusters com k={best_k_silhouette} (Silhueta)')

plt.show()
