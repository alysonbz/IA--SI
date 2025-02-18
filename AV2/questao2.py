import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

df = pd.read_csv(r"C:\Users\bende\av1\IA--SI\AV2\star_classification.csv")

df_sample = df.sample(frac=0.1, random_state=42)

X = df_sample.drop(columns=['class']).values

X = StandardScaler().fit_transform(X)

def calculate_kmeans_metrics(k):
    kmeans = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=100, max_iter=100, tol=1e-4)
    kmeans.fit(X)

    sse = kmeans.inertia_

    silhouette = silhouette_score(X, kmeans.labels_) if k > 1 else np.nan

    return k, sse, silhouette

cluster_range = range(2, 7)

metrics = [calculate_kmeans_metrics(k) for k in cluster_range]

sse = [metric[1] for metric in metrics]
silhouette_scores = [metric[2] for metric in metrics]

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(cluster_range, sse, marker='o')
plt.title('Método do Cotovelo')
plt.xlabel('Número de Clusters')
plt.ylabel('Soma dos Erros Quadráticos (SSE)')
plt.xticks(cluster_range)
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(cluster_range, silhouette_scores, marker='o', color='green')
plt.title('Método da Silhueta')
plt.xlabel('Número de Clusters')
plt.ylabel('Pontuação da Silhueta')
plt.xticks(cluster_range)
plt.grid(True)

plt.tight_layout()
plt.show()

for k, score in zip(cluster_range, silhouette_scores):
    print(f"{k} clusters: {score:.4f}")
