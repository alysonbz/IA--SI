import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Lasso
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


caminho_arquivo = r"C:\Users\vitor\Downloads\IA.BLACK\IA--SI\AV2\star_classification.csv"
df = pd.read_csv(caminho_arquivo)


df_sample = df.sample(frac=0.05, random_state=42)  # Pegando 5% dos dados


colunas_relevantes = [
    'alpha', 'delta', 'u', 'g', 'r', 'i', 'z',
    'redshift', 'plate', 'MJD'
]
df_sample_final = df_sample[colunas_relevantes].copy()  # Criando uma cópia independente


scaler = StandardScaler()
X = scaler.fit_transform(df_sample_final)


lasso = Lasso(alpha=0.1)
lasso.fit(X, X)  # Ajustando com X como target (uma técnica de seleção de features não supervisionada)


coef = np.abs(lasso.coef_)


relevantes_idx = coef.argsort()[-2:][::-1]  # Pegando os dois maiores coeficientes
relevantes_idx = relevantes_idx.flatten()  # Garantindo que seja unidimensional

atributos_relevantes = df_sample_final.columns[relevantes_idx].tolist()  # Pegando os atributos diretamente da coluna

print(f"Atributos mais relevantes selecionados pelo Lasso: {atributos_relevantes}")


X_relevantes = X[:, relevantes_idx]  # Pegando apenas os dois atributos mais relevantes


pca = PCA(n_components=2)  # Reduzir para 2 componentes principais para visualização
X_pca = pca.fit_transform(X_relevantes)


silhouette_scores = []
for k in range(2, 6):  # Diminuindo o intervalo de busca para acelerar o processo
    kmeans = MiniBatchKMeans(n_clusters=k, random_state=42, max_iter=200)  # Usando MiniBatchKMeans
    kmeans.fit(X_pca)
    score = silhouette_score(X_pca, kmeans.labels_)
    silhouette_scores.append(score)


k_silhueta = np.argmax(silhouette_scores) + 2  # Adicionamos 2 porque a contagem começa de 2


kmeans = MiniBatchKMeans(n_clusters=k_silhueta, random_state=42, max_iter=200)
clusters = kmeans.fit_predict(X_pca)


df_sample_final.loc[:, 'cluster'] = clusters


df_sample_final.loc[:, 'class'] = df_sample['class'].values


crosstab = pd.crosstab(df_sample_final['class'], df_sample_final['cluster'], rownames=['Classe'], colnames=['Cluster'])


print("\nDistribuição de Clusters de Acordo com as Classes:")
print(crosstab)


plt.figure(figsize=(8, 6))
sns.heatmap(crosstab, annot=True, cmap='Blues', fmt='d', cbar=False)
plt.title('Distribuição de Clusters de Acordo com as Classes')
plt.show()
