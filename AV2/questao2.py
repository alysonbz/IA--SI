import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Carregar o dataset
classificacao = pd.read_csv('./dataset/dataset_classificacao_ajustado.csv')

# Remover a coluna alvo
X = classificacao.drop(['booking_status'], axis=1)

# Normalização dos dados
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Método do Cotovelo - Determinar o melhor K
inertia = []

for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# Determinar o melhor k com o método do cotovelo
best_k_elbow = 8

# Plot do gráfico - Método do Cotovelo
plt.plot(range(1, 11), inertia, marker='o')
plt.title('Método do Cotovelo')
plt.xlabel('Número de Clusters (K)')
plt.ylabel('Inertia')
plt.show()

# OBS:
# Os metódos são rodados separadamente, primeiro o Cotovelo e só depois o Shilhouette, para não haver sobrercarga no computador.
# Por isso é comentado o metódo que não vai ser usado na análise.

# # Metódo da Silhouette
# silhouette_scores = []
#
# for k in range(2, 11):
#     kmeans = KMeans(n_clusters=k, random_state=42)
#     labels = kmeans.fit_predict(X_scaled)
#     score = silhouette_score(X_scaled, labels)
#     silhouette_scores.append(score)
#
# # Determinar o melhor k com o método de Silhueta
# best_k_silhouette = silhouette_scores.index(max(silhouette_scores)) + 2  # +2 porque começamos com k=2
#
# # Plot do Gráfico da Silhueta
# plt.figure(figsize=(8, 6))
# plt.plot(range(2, 11), silhouette_scores, marker='o')
# plt.title('Índice de Silhueta para Diferentes Valores de K')
# plt.xlabel('Número de Clusters (K)')
# plt.ylabel('Silhouette Score')
# plt.show()