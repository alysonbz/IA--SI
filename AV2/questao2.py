import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import silhouette_score

# Carregar o dataset
data = pd.read_csv("drug200.csv")

# Codificar variáveis categóricas usando LabelEncoder
label_encoders = {}
for column in ['Sex', 'BP', 'Cholesterol', 'Drug']:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le  # Salvar o encoder para uso posterior, se necessário

# Remover a coluna alvo 'Drug'
data_numeric = data.drop(columns=['Drug'])

# Padronizar os dados (importante para clustering)
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_numeric)

# Lista para armazenar a Inertia para diferentes valores de K
inertia = []

# Testar para diferentes valores de K (de 1 a 10)
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(data_scaled)
    inertia.append(kmeans.inertia_)
    print(f"Inertia para k={k}: {kmeans.inertia_}")

# Plotar o gráfico do método do cotovelo
plt.figure(figsize=(8, 6))
plt.plot(range(1, 11), inertia, marker='o')
plt.title('Método do Cotovelo')
plt.xlabel('Número de Clusters (K)')
plt.ylabel('Inertia')
plt.grid(True)
plt.show()

# Método da Silhueta
silhouette_scores = []  # Lista para armazenar os scores de silhueta
for k in range(2, 11):  # O score de silhueta requer pelo menos 2 clusters
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(data_scaled)
    score = silhouette_score(data_scaled, kmeans.labels_)
    silhouette_scores.append(score)

# Plotar o gráfico do método da silhueta
plt.figure(figsize=(8, 6))
plt.plot(range(2, 11), silhouette_scores, marker='o')
plt.title('Índice de Silhueta para Diferentes Valores de K')
plt.xlabel('Número de Clusters (K)')
plt.ylabel('Score de Silhueta')
plt.show()