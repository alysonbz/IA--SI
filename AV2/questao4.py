import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Lasso
import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

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

# Armazenar as pontuações de silhueta para cada k
best_silhouette_score = -1  # Inicializa com valor baixo
best_k = 2

# Para cada valor de k, calcular e armazenar a pontuação de silhueta
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_selected)

    # Calcular a pontuação de silhueta para cada amostra
    silhouette_vals = silhouette_samples(X_selected, cluster_labels)

    # Calcular a média da pontuação de silhueta
    avg_silhouette_score = np.mean(silhouette_vals)

    # Atualizar a melhor pontuação de silhueta e o valor de k
    if avg_silhouette_score > best_silhouette_score:
        best_silhouette_score = avg_silhouette_score
        best_k = k

# Aplicar o KMeans com o melhor k
kmeans_best = KMeans(n_clusters=best_k, random_state=42, n_init=10)
cluster_labels_best = kmeans_best.fit_predict(X_selected)

# Criar o crosstab entre clusters e a variável alvo
crosstab = pd.crosstab(cluster_labels_best, y)
print("Crosstab da Distribuição de Clusters e Classes Alvo:")
print(crosstab)
