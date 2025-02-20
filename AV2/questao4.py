import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

dataset_path = "Dataset_coletado.csv"
df = pd.read_csv(dataset_path)

df = df.dropna()

coluna_alvo = "blue"  
X = df.drop(columns=["blue"])
y = df["blue"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

range_k = range(2, 11)
best_k = 2
best_score = -1

for k in range_k:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    
    if score > best_score:
        best_score = score
        best_k = k

print(f"Melhor valor de k pelo índice de silhueta: {best_k} (score = {best_score:.4f})")

kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(X_scaled)

crosstab_result = pd.crosstab(df["Cluster"], df[coluna_alvo])
print("\nDistribuição de clusters por classe na coluna-alvo:")
print(crosstab_result)

crosstab_result.to_csv("crosstab_clusters.csv")
print("\nCrosstab salva em 'crosstab_clusters.csv'.")