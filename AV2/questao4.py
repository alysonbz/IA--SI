import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Lasso
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

caminho_arquivo = r"C:\Users\bende\av1\IA--SI\AV2\star_classification.csv"
df = pd.read_csv(caminho_arquivo)

df_sample = df.sample(frac=0.05, random_state=42)  # Pegando 5% dos dados

colunas_relevantes = [
    'alpha', 'delta', 'u', 'g', 'r', 'i', 'z',
    'redshift', 'plate', 'MJD'
]
df_sample_final = df_sample[colunas_relevantes]

scaler = StandardScaler()
X = scaler.fit_transform(df_sample_final)

lasso = Lasso(alpha=0.1)
lasso.fit(X, X)

coef = np.abs(lasso.coef_)

relevantes_idx = coef.argsort()[-2:][::-1]
relevantes_idx = relevantes_idx.flatten()

atributos_relevantes = df_sample_final.columns[relevantes_idx].tolist()
print(f"Atributos mais relevantes selecionados pelo Lasso: {atributos_relevantes}")

X_relevantes = X[:, relevantes_idx]
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_relevantes)

silhouette_scores = []
for k in range(2, 6):
    kmeans = MiniBatchKMeans(n_clusters=k, random_state=42, max_iter=200)
    kmeans.fit(X_pca)
    score = silhouette_score(X_pca, kmeans.labels_)
    silhouette_scores.append(score)

k_silhueta = np.argmax(silhouette_scores) + 2

kmeans = MiniBatchKMeans(n_clusters=k_silhueta, random_state=42, max_iter=200)
clusters = kmeans.fit_predict(X_pca)

df_sample_final['cluster'] = clusters

df_sample_final['class'] = df_sample['class']

crosstab = pd.crosstab(df_sample_final['class'], df_sample_final['cluster'], rownames=['Classe'], colnames=['Cluster'])

print("\nDistribuição de Clusters de Acordo com as Classes:")
print(crosstab)

plt.figure(figsize=(8, 6))
sns.heatmap(crosstab, annot=True, cmap='Blues', fmt='d', cbar=False)
plt.title('Distribuição de Clusters de Acordo com as Classes')
plt.show()
