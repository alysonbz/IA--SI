import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Carregar o dataset
df = pd.read_csv('waterquality_ajustado.csv')

# Visualizar as primeiras linhas do dataset
print(df.head())

# Supondo que a coluna alvo seja 'is_safe' e as variáveis independentes sejam as outras colunas
X = df.drop('is_safe', axis=1)  # Remover a coluna alvo
y = df['is_safe']  # A coluna alvo

# Normalizar os dados
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Determinação do número de clusters com o método da silhueta
silhouette_scores = []
for k in range(2, 11):  # Testando k de 2 a 10
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))

# Encontrar o melhor valor de k (com maior pontuação de silhueta)
k_silhueta = silhouette_scores.index(max(silhouette_scores))  # k começa de 2
print(f"O melhor número de clusters (k) pelo método da silhueta é: {k_silhueta}")

# Aplicar KMeans com o k obtido
kmeans = KMeans(n_clusters=k_silhueta, random_state=42)
df['cluster'] = kmeans.fit_predict(X_scaled)

# Criar a tabela de contingência (crosstab)
crosstab = pd.crosstab(df['cluster'], df['is_safe'], margins=True)
print(crosstab)

# Exibir um gráfico para visualizar a distribuição dos clusters
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=df['cluster'], cmap='viridis')
plt.title(f'Clusters com K = {k_silhueta}')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()
