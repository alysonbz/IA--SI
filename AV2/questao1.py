import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# 1) Carregar o dataset
df = pd.read_csv('waterquality_ajustado.csv')

# 2) Definir variáveis preditoras (X) e variável alvo (y)
X = df.drop(columns=['is_safe'])
y = df['is_safe']

# 3) Dividir os dados em treino (80%) e teste (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 4) Configurar os parâmetros para GridSearchCV
param_grid = {
    'n_neighbors': range(1, 21),  # Testar k de 1 a 20
    'metric': ['euclidean', 'manhattan', 'chebyshev'],  # Testar diferentes métricas
    'weights': ['uniform', 'distance']  # Testar pesos uniformes e por distância
}

# 5) Criar o modelo KNN
knn = KNeighborsClassifier()

# 6) Aplicar GridSearchCV para encontrar os melhores parâmetros
grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

# 7) Exibir os melhores parâmetros encontrados
print("Melhores Parâmetros:", grid_search.best_params_)

# 8) Avaliar o modelo com os melhores parâmetros no conjunto de teste
best_knn = grid_search.best_estimator_
y_pred = best_knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nAcurácia no conjunto de teste: {accuracy:.4f}")
#AV1euclidean: 8774 com KNN 7
#AV2euclidean: 8793 com KNN: 18