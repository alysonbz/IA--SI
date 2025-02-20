import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# 1) Carregar o dataset
df = pd.read_csv('waterquality_ajustado.csv')

# 2) Definir as variáveis preditoras (remover a coluna alvo 'is_safe')
X = df.drop(columns=['is_safe'])

# 3) Método do Cotovelo
# Vamos calcular a soma dos erros quadráticos dentro dos clusters (inertia) para diferentes valores de k
inertia = []
silhueta_score = []

for k in range(2, 11):  # Testando de k=2 até k=10 de clusters
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    # Soma dos erros quadráticos dentro dos clusters
    inertia.append(kmeans.inertia_)
    # Calculando o coeficiente da silhueta para o modelo com k clusters
    score = silhouette_score(X, kmeans.labels_)
    silhueta_score.append(score)

# 4) Método do Cotovelo: Plotando a inércia (soma dos erros quadráticos dentro dos clusters)
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(range(1, 10), inertia, marker='o')
plt.title('Método do Cotovelo')
plt.xlabel('Número de Clusters (k)')
plt.ylabel('Inércia')

# 5) Método da Silhueta: Plotando o coeficiente de silhueta para diferentes k
plt.subplot(1, 2, 2)
plt.plot(range(2, 11), silhueta_score, marker='o')
plt.title('Método da Silhueta')
plt.xlabel('Número de Clusters (k)')
plt.ylabel('Coeficiente da Silhueta')

plt.tight_layout()
plt.show()

# 6) Exibindo o melhor número de clusters baseado no método do cotovelo e da silhueta
best_k_silhouette = range(2, 11)[np.argmax(silhueta_score)]
best_k_cotovelo = range(2,11)[np.argmax(inertia)]
print(f'a melhor silhueta {best_k_silhouette}')
print(f'o melhor cotovelo {best_k_cotovelo}')