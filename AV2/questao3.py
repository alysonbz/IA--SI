import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# 1. Carregar o dataset
df = pd.read_csv('waterquality_ajustado.csv')

# 2. Definir as variáveis preditoras (X) e a variável alvo (y)
X = df.drop(columns=['is_safe'])  # Aqui você remove a coluna alvo 'is_safe'
y = df['is_safe']  # A variável alvo

# 3. Ajustar o modelo Lasso para selecionar as variáveis mais relevantes
lasso = Lasso(alpha=0.1)  # O valor de alpha pode ser ajustado
lasso.fit(X, y)

# 4. Obter os coeficientes das variáveis e selecionar os dois mais relevantes
coef = lasso.coef_
coef_df = pd.DataFrame({'Feature': X.columns, 'Coefficient': coef})
coef_df = coef_df.sort_values(by='Coefficient', ascending=False)

# Selecionar os dois atributos com maiores coeficientes
top_features = coef_df['Feature'].head(2).values
print(f'Os dois atributos mais relevantes são: {top_features}')

# 5. Filtrar os dados para usar apenas os dois atributos mais relevantes
X_reduced = X[top_features]

# 6. Método do Cotovelo para determinar o número ideal de clusters
inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_reduced)
    inertia.append(kmeans.inertia_)

# Plotar o gráfico do cotovelo
plt.plot(range(1, 11), inertia, marker='o')
plt.title('Método do Cotovelo (após seleção Lasso)')
plt.xlabel('Número de Clusters (k)')
plt.ylabel('Inércia')
plt.show()

# 7. Método da Silhueta para determinar o número ideal de clusters
silhouette_scores = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_reduced)
    score = silhouette_score(X_reduced, kmeans.labels_)
    silhouette_scores.append(score)

# Plotar o gráfico da silhueta
plt.plot(range(2, 11), silhouette_scores, marker='o')
plt.title('Método da Silhueta (após seleção Lasso)')
plt.xlabel('Número de Clusters (k)')
plt.ylabel('Pontuação de Silhueta')
plt.show()

# 8. Criar os KMeans com o número de clusters obtido
k_cotovelo = 4
kmeans_cotovelo = KMeans(n_clusters=k_cotovelo, random_state=42)
labels_cotovelo = kmeans_cotovelo.fit_predict(X_reduced)

k_silhueta = 4
kmeans_silhueta = KMeans(n_clusters=k_silhueta, random_state=42)
labels_silhueta = kmeans_silhueta.fit_predict(X_reduced)

# 9. Plotar os scatterplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Scatterplot para o Método do Cotovelo
ax1.scatter(X_reduced.iloc[:, 0], X_reduced.iloc[:, 1], c=labels_cotovelo, cmap='viridis')
ax1.set_title(f'Clusters (Método do Cotovelo - k={k_cotovelo})')
ax1.set_xlabel(top_features[0])
ax1.set_ylabel(top_features[1])

# Scatterplot para o Método da Silhueta
ax2.scatter(X_reduced.iloc[:, 0], X_reduced.iloc[:, 1], c=labels_silhueta, cmap='viridis')
ax2.set_title(f'Clusters (Método da Silhueta - k={k_silhueta})')
ax2.set_xlabel(top_features[0])
ax2.set_ylabel(top_features[1])

plt.tight_layout()
plt.show()