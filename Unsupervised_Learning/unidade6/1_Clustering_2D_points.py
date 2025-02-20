from src.utils import load_points
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Carregar os pontos do dataset
points = load_points()

# Criar uma instância do KMeans com 3 clusters
model = KMeans(n_clusters=3, random_state=42)

# Dividir os dados em treino e teste
test_points = points[:50, :]
train_points = points[50:, :]

# Treinar o modelo com os pontos de treinamento
model.fit(train_points)

# Prever os rótulos dos pontos de teste
labels = model.predict(test_points)

# Imprimir os rótulos dos clusters atribuídos aos pontos de teste
print(labels)

# Separar coordenadas X e Y dos pontos de teste
xs = test_points[:, 0]
ys = test_points[:, 1]

# Criar um gráfico de dispersão dos pontos de teste, colorindo pelos rótulos do cluster
plt.scatter(xs, ys, c=labels, cmap='viridis', alpha=0.5)

# Obter os centróides dos clusters
centroids = model.cluster_centers_

# Separar coordenadas X e Y dos centróides
centroids_x = centroids[:, 0]
centroids_y = centroids[:, 1]

# Adicionar os centróides ao gráfico
plt.scatter(centroids_x, centroids_y, s=100, marker='D', color='red', label='Centroids')

# Mostrar o gráfico
plt.legend()
plt.show()
