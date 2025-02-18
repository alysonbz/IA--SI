import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from joblib import Parallel, delayed

# Carregar e preparar os dados
df = pd.read_csv(r"C:\Users\bende\av1\IA--SI\AV2\star_classification.csv")

# Amostragem do dataset (reduzido a 10% para otimização)
df_sample = df.sample(frac=0.1, random_state=42)

# Normalização
X = StandardScaler().fit_transform(df_sample[['alpha', 'delta', 'u', 'g', 'r', 'i', 'z', 'redshift', 'plate', 'MJD']])

# 1. Seleção de Atributos com Lasso
lasso = Lasso(alpha=0.1).fit(X, X)
relevantes_idx = np.abs(lasso.coef_).argsort()[-2:][::-1]  # Obtenha os índices dos dois atributos mais relevantes

# Garantir que 'relevantes_idx' seja um vetor de inteiros unidimensionais
relevantes_idx = relevantes_idx.flatten()  # Certifique-se de que seja unidimensional

# Usando esses índices para selecionar os atributos relevantes corretamente
atributos_relevantes = df.columns[relevantes_idx].tolist()  # Acessando as colunas corretamente
print(f"Atributos mais relevantes selecionados pelo Lasso: {atributos_relevantes}")

# 2. Recalcular a Quantidade de Clusters com Método do Cotovelo e Silhueta
X_relevantes = X[:, relevantes_idx]  # Selecionando apenas as colunas relevantes

# Redução de dimensionalidade com PCA (se necessário)
pca = PCA(n_components=2)  # Reduzir para 2 componentes principais para visualização
X_pca = pca.fit_transform(X_relevantes)


# Função para calcular o SSE e a Silhueta para MiniBatchKMeans
def calculate_kmeans_metrics(k):
    kmeans = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=100)
    kmeans.fit(X_pca)

    sse = kmeans.inertia_  # SSE
    silhouette = silhouette_score(X_pca, kmeans.labels_) if k > 1 else np.nan

    return k, sse, silhouette


# Paralelizando os cálculos de SSE e Silhueta para 2 até 3 clusters (limite para reduzir o uso de recursos)
metrics = Parallel(n_jobs=2)(
    delayed(calculate_kmeans_metrics)(k) for k in range(2, 4))  # Alterei para n_jobs=2 e clusters 2-3

# Separando os resultados de SSE e Silhueta para plotagem
sse = [metric[1] for metric in metrics]
silhouette_scores = [metric[2] for metric in metrics]

# Plotando o método do Cotovelo
plt.plot(range(2, 4), sse, marker='o')
plt.title('Método do Cotovelo')
plt.xlabel('Número de Clusters')
plt.ylabel('Soma dos Erros Quadráticos (SSE)')
plt.xticks(range(2, 4))
plt.grid(True)
plt.show()

# Plotando o método da Silhueta
plt.plot(range(2, 4), silhouette_scores, marker='o', color='green')
plt.title('Método da Silhueta')
plt.xlabel('Número de Clusters')
plt.ylabel('Pontuação da Silhueta')
plt.xticks(range(2, 4))
plt.grid(True)
plt.show()

# Exibir pontuação da silhueta
for k, score in zip(range(2, 4), silhouette_scores):
    print(f"{k} clusters: {score:.4f}")

# 3. Visualização com Scatterplots (k=3 como exemplo)
kmeans = MiniBatchKMeans(n_clusters=3, random_state=42, batch_size=100)
labels = kmeans.fit_predict(X_pca)

# Scatterplot
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis')
plt.title(f'Clusters com 3 clusters')
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.colorbar(label='Cluster')
plt.show()
