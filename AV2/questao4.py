import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

df = pd.read_csv(r"C:\Users\Administrator\IA--SI\AV2\healthcare-dataset-stroke-data.csv")

atributos_relevantes = ["age", "hypertension", "heart_disease", "avg_glucose_level", "bmi"]
print("Valores ausentes antes do tratamento:\n", df[atributos_relevantes].isna().sum())

df = df.dropna(subset=atributos_relevantes)
print("\nValores ausentes após o tratamento:\n", df[atributos_relevantes].isna().sum())

X = df[atributos_relevantes]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

k = 3
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(X_scaled)

crosstab_result = pd.crosstab(df["Cluster"], df["stroke"])
print("\nTabela de Contingência (Crosstab):")
print(crosstab_result)

plt.figure(figsize=(10, 6))
sns.scatterplot(x=df["age"], y=df["avg_glucose_level"], hue=df["Cluster"], palette="viridis", s=100, alpha=0.7)

centroids = kmeans.cluster_centers_
for i, centroid in enumerate(centroids):
    plt.scatter(centroid[0], centroid[3], marker='X', s=200, c='red', label=f'Centroid {i+1}' if i == 0 else "")

plt.title('Distribuição dos Pacientes com Base nos Clusters (Age vs Avg Glucose Level) com Centroids', fontsize=16)
plt.xlabel('Age', fontsize=12)
plt.ylabel('Avg Glucose Level', fontsize=12)
plt.legend(title='Cluster', loc='upper right')
plt.show()