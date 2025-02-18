import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


df = pd.read_csv(r'AV2\Cancer_Data.csv')
print(df.columns)

df = df.drop(columns=["id", "Unnamed: 32"], errors='ignore')

df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})

X = df.drop(columns=["diagnosis"])
y = df["diagnosis"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

lasso = Lasso(alpha=0.01)
lasso.fit(X_scaled, y)

# Identificar os dois atributos mais relevantes
coeficientes = np.abs(lasso.coef_)
atributos_importantes = np.argsort(coeficientes)[-2:]
atributos_nomes = X.columns[atributos_importantes]
X_lasso = X.iloc[:, atributos_importantes]

# Imprimir os dois atributos mais importantes
print('Atributos mais relevantes:')
print(atributos_nomes.tolist())

# Método do Cotovelo
iner = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_lasso)
    iner.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(range(2, 11), iner, marker='o', linestyle='--')
plt.xlabel("Número de Clusters (k)")
plt.ylabel("WCSS")
plt.title("Método do Cotovelo")
plt.show()

# Método da Silhueta
silhouette_scores = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_lasso)
    silhouette_scores.append(silhouette_score(X_lasso, cluster_labels))

plt.figure(figsize=(8, 5))
plt.plot(range(2, 11), silhouette_scores, marker='o', linestyle='--')
plt.xlabel("Número de Clusters (k)")
plt.ylabel("Silhouette Score")
plt.title("Método da Silhueta")
plt.show()

print(df.info())