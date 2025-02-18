import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.linear_model import Lasso

# Carregar dataset
df = pd.read_csv(r"C:\Users\gabriel\Downloads\archive\gender_classification_v7.csv")
# Converter a coluna alvo para valores numéricos
le = LabelEncoder()
df["gender"] = le.fit_transform(df["gender"])

# Separar features e alvo
X = df.drop(columns=["gender"])
y = df["gender"]

# Normalizar os dados
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Seleção de atributos usando Lasso
lasso = Lasso(alpha=0.001)  # Ajuste menor para evitar coeficientes zerados
lasso.fit(X_scaled, y)
feature_importance = np.abs(lasso.coef_)
selected_features = np.argsort(feature_importance)[-2:]
X_selected = X_scaled[:, selected_features]

# Método do Cotovelo com os atributos selecionados
inertia = []
k_range = range(1, 11)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_selected)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(k_range, inertia, marker='o')
plt.xlabel('Número de Clusters')
plt.ylabel('Inércia')
plt.title('Método do Cotovelo (Atributos Selecionados)')
plt.show()

# Método da Silhueta com os atributos selecionados
silhouette_scores = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_selected)
    silhouette_scores.append(silhouette_score(X_selected, labels))

plt.figure(figsize=(8, 5))
plt.plot(range(2, 11), silhouette_scores, marker='o')
plt.xlabel('Número de Clusters')
plt.ylabel('Coeficiente de Silhueta')
plt.title('Método da Silhueta (Atributos Selecionados)')
plt.show()

# Scatterplots para visualização dos clusters
best_k_elbow = 3  # Ajuste com base na análise visual
best_k_silhouette = 4

kmeans_elbow = KMeans(n_clusters=best_k_elbow, random_state=42, n_init=10)
kmeans_silhouette = KMeans(n_clusters=best_k_silhouette, random_state=42, n_init=10)

labels_elbow = kmeans_elbow.fit_predict(X_selected)
labels_silhouette = kmeans_silhouette.fit_predict(X_selected)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(X_selected[:, 0], X_selected[:, 1], c=labels_elbow, cmap='viridis')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title(f'Clusters pelo Método do Cotovelo (k={best_k_elbow})')

plt.subplot(1, 2, 2)
plt.scatter(X_selected[:, 0], X_selected[:, 1], c=labels_silhouette, cmap='viridis')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title(f'Clusters pelo Método da Silhueta (k={best_k_silhouette})')

plt.show()

# Crosstab para distribuição de clusters pelo índice de silhueta
kmeans_final = KMeans(n_clusters=best_k_silhouette, random_state=42, n_init=10)
df['cluster'] = kmeans_final.fit_predict(X_selected)
crosstab_result = pd.crosstab(df['cluster'], df['gender'])
print("Distribuição de Clusters por Classe:")
print(crosstab_result)

# Dividir em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Definir os parâmetros para o GridSearchCV
param_grid = {
    'n_neighbors': range(1, 21),
    'metric': ['euclidean', 'manhattan', 'minkowski']
}

# Criar o modelo KNN e realizar a busca
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
