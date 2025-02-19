import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4"
# Carregar o dataset
data = pd.read_csv("drug200.csv")

# Codificar variáveis categóricas usando LabelEncoder
label_encoders = {}
for column in ['Sex', 'BP', 'Cholesterol', 'Drug']:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le  # Salvar o encoder para uso posterior, se necessário

# Separar os dados e o target
X = data.drop(columns=['Drug'])  # Features
y = data['Drug']  # Target

# Padronizar os dados para o Lasso
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Aplicar Lasso
lasso = Lasso(alpha=0.01)  # alpha controla a força da regularização
lasso.fit(X_scaled, y)

# Obter os coeficientes das variáveis
feature_importance = pd.Series(lasso.coef_, index=X.columns)
print("Importância das variáveis pelo Lasso:")
print(feature_importance)

# Selecionar as duas variáveis mais relevantes (coeficientes não nulos)
selected_features = feature_importance.abs().nlargest(2).index
print("\nAs duas variáveis mais relevantes são:", selected_features)

# Filtrar apenas as variáveis selecionadas
X_selected = X[selected_features]

# Padronizar os dados novamente
X_selected_scaled = scaler.fit_transform(X_selected)

# Método do Cotovelo
inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_selected_scaled)
    inertia.append(kmeans.inertia_)

# Plotar o gráfico do método do cotovelo
plt.figure(figsize=(8, 6))
plt.plot(range(1, 11), inertia, marker='o')
plt.title('Método do Cotovelo (Apenas Variáveis Relevantes)')
plt.xlabel('Número de Clusters (K)')
plt.ylabel('Inertia')
plt.grid(True)
plt.show()

# Método da Silhueta
silhouette_scores = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_selected_scaled)
    score = silhouette_score(X_selected_scaled, kmeans.labels_)
    silhouette_scores.append(score)

# Plotar o gráfico do método da silhueta
plt.figure(figsize=(8, 6))
plt.plot(range(2, 11), silhouette_scores, marker='o')
plt.title('Índice de Silhueta (Apenas Variáveis Relevantes)')
plt.xlabel('Número de Clusters (K)')
plt.ylabel('Score de Silhueta')
plt.show()