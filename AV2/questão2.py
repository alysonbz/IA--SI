import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Carregar dataset
df = pd.read_csv(r"C:\Users\gabriel\Downloads\archive\gender_classification_v7.csv")

# Remover a coluna alvo para análise de clusters
X = df.drop(columns=["gender"])

# Normalizar os dados
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Método do Cotovelo
inertia = []
k_range = range(1, 11)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(k_range, inertia, marker='o')
plt.xlabel('Número de Clusters')
plt.ylabel('Inércia')
plt.title('Método do Cotovelo')
plt.show()

# Método da Silhueta
silhouette_scores = []
for k in range(2, 11):  # Silhueta não é definida para k=1
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    silhouette_scores.append(silhouette_score(X_scaled, labels))

plt.figure(figsize=(8, 5))
plt.plot(range(2, 11), silhouette_scores, marker='o')
plt.xlabel('Número de Clusters')
plt.ylabel('Coeficiente de Silhueta')
plt.title('Método da Silhueta')
plt.show()

# iremos dividir em treino e testee
X_train, X_test, y_train, y_test = train_test_split(X, df["gender"], test_size=0.2, random_state=42)

# aqui estaremos normalizando os dados
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# definindo os parametros pra GridSearchCV
param_grid = {
    'n_neighbors': range(1, 21),  # Testando valores de k de 1 a 20
    'metric': ['euclidean', 'manhattan', 'minkowski']  # Testando diferentes métricas de distância
}

# criando o knn e fazendo a busca
grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Melhor combinação de parâmetros
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

# Avaliar no conjunto de teste
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Melhores parâmetros:", best_params)
print("Acurácia no conjunto de teste:", accuracy)
print("Relatório de classificação:\n", report)
