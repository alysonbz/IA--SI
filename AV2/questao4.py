import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Lasso
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Carregar os dados (Amostragem de 5% para evitar sobrecarga)
caminho_arquivo = r"C:\Users\bende\av1\IA--SI\AV2\star_classification.csv"
df = pd.read_csv(caminho_arquivo)

# Amostragem de 5% para reduzir o tamanho do dataset
df_sample = df.sample(frac=0.05, random_state=42)  # Pegando 5% dos dados

# Selecionar as colunas relevantes, exceto a coluna 'class'
colunas_relevantes = [
    'alpha', 'delta', 'u', 'g', 'r', 'i', 'z',
    'redshift', 'plate', 'MJD'
]
df_sample_final = df_sample[colunas_relevantes]

# Normalizar os dados
scaler = StandardScaler()
X = scaler.fit_transform(df_sample_final)

# 1. Seleção de Atributos com Lasso
lasso = Lasso(alpha=0.1)
lasso.fit(X, X)  # Ajustando com X como target (uma técnica de seleção de features não supervisionada)

# Obter os coeficientes
coef = np.abs(lasso.coef_)

# Selecionar os dois atributos mais relevantes (maiores coeficientes)
relevantes_idx = coef.argsort()[-2:][::-1]  # Pegando os dois maiores coeficientes
relevantes_idx = relevantes_idx.flatten()  # Garantindo que seja unidimensional

atributos_relevantes = df_sample_final.columns[relevantes_idx].tolist()  # Agora pegamos os atributos diretamente da coluna

print(f"Atributos mais relevantes selecionados pelo Lasso: {atributos_relevantes}")

# 2. Recalcular a Quantidade de Clusters com Método da Silhueta e MiniBatchKMeans
X_relevantes = X[:, relevantes_idx]  # Aqui garantimos que estamos pegando apenas os dois atributos mais relevantes

# Redução de Dimensionalidade (PCA)
pca = PCA(n_components=2)  # Reduzir para 2 componentes principais para visualização
X_pca = pca.fit_transform(X_relevantes)

# 2.1 Método da Silhueta (Silhouette Method) com um número restrito de clusters (de 2 a 5)
silhouette_scores = []
for k in range(2, 6):  # Diminuindo o intervalo de busca para acelerar o processo
    kmeans = MiniBatchKMeans(n_clusters=k, random_state=42, max_iter=200)  # Usando MiniBatchKMeans
    kmeans.fit(X_pca)
    score = silhouette_score(X_pca, kmeans.labels_)
    silhouette_scores.append(score)

# Encontrando o valor de k ótimo com base no método da silhueta
k_silhueta = np.argmax(silhouette_scores) + 2  # Adicionamos 2 porque a contagem começa de 2

# 3. Aplicar o MiniBatchKMeans com o k ótimo
kmeans = MiniBatchKMeans(n_clusters=k_silhueta, random_state=42, max_iter=200)
clusters = kmeans.fit_predict(X_pca)

# Adicionar as previsões de cluster ao DataFrame original (para fácil visualização)
df_sample_final['cluster'] = clusters

# Adicionar a coluna 'class' ao DataFrame para facilitar a comparação
df_sample_final['class'] = df_sample['class']

# 4. Criar o crosstab
crosstab = pd.crosstab(df_sample_final['class'], df_sample_final['cluster'], rownames=['Classe'], colnames=['Cluster'])

# Exibir o crosstab
print("\nDistribuição de Clusters de Acordo com as Classes:")
print(crosstab)

# 5. Visualizar a distribuição de clusters em relação às classes
plt.figure(figsize=(8, 6))
sns.heatmap(crosstab, annot=True, cmap='Blues', fmt='d', cbar=False)
plt.title('Distribuição de Clusters de Acordo com as Classes')
plt.show()
