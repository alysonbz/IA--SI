import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Lasso

# Carregar o dataset
df = pd.read_csv("sabores_de_cacau_ajustado.csv")

# Separar features e target
if 'rating' in df.columns:
    X = df.drop(columns=['rating'])
    y = df['rating']
else:
    X = df.copy()
    y = None

# Verifica se existem valores faltantes
X.fillna(X.mean(), inplace=True)  # Substitui NaNs pela média

# Normalizar os dados
scaler = MinMaxScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Aplicar Lasso para selecionar os atributos mais relevantes
lasso = Lasso(alpha=0.01)
lasso.fit(X_scaled, y)

importance = np.abs(lasso.coef_)

# Evita erro se todas as importâncias forem zero
if np.all(importance == 0):
    top_2_features = X.columns[:2]  # Pega as duas primeiras colunas como fallback
else:
    top_2_features = X.columns[np.argsort(importance)[-2:]]

X_selected = X_scaled[top_2_features]

# Lista de valores de k para testar
k_values = list(range(2, 11))

# Criar uma figura para o gráfico de silhueta
plt.figure(figsize=(12, 8))

# Para cada valor de k, calcular e plotar o gráfico de silhueta
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_selected)

    # Calcular as pontuações de silhueta para cada amostra
    silhouette_vals = silhouette_samples(X_selected, cluster_labels)

    # Plotar o gráfico de silhueta
    plt.subplot(2, 5, k - 1)
    plt.hist(silhouette_vals, bins=30, edgecolor='black')
    plt.title(f'Silhueta para k={k}')
    plt.xlabel('Pontuação de Silhueta')
    plt.ylabel('Número de Amostras')
    plt.xlim([-1, 1])
    plt.tight_layout()

# Exibir os gráficos
plt.show()
