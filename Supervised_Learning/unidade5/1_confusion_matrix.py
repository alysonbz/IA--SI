from src.utils import load_diabetes_clean_dataset
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix

diabetes_df = load_diabetes_clean_dataset()
X = diabetes_df.drop(['diabetes'],axis=1)
y = diabetes_df['diabetes'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42,stratify=y)

knn = KNeighborsClassifier(n_neighbors=6)

# Fit the model to the training data
# knn.fit(X_train, y_train)

# Atividade extra: melhorar o knn utilizando o GridSearchCV

# Definindo hiperparâmetros
param_grid = {
    'n_neighbors': [3, 5, 7, 9, 11],
    'metric': ['euclidean', 'manhattan', 'chebyshev']
}

# Criando o GridSearchCV para encontrar os melhores parâmetros
grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy', n_jobs=-1)

# Ajustar o modelo aos dados de treino usando GridSearchCV
grid_search.fit(X_train, y_train)

# Obter os melhores hiperparâmetros encontrados
best_params = grid_search.best_params_
print(f"Melhores parâmetros encontrados: {best_params}")

# Utilizar o melhor modelo encontrado pelo GridSearchCV
best_knn = grid_search.best_estimator_

# Fazer previsões no conjunto de teste
y_pred = best_knn.predict(X_test)

# Exibir métricas de avaliação
print("Matriz de Confusão:")
print(confusion_matrix(y_test, y_pred))
print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred))

# Generate the confusion matrix and classification report
# print(confusion_matrix(y_test, y_pred))
# print(classification_report(y_test, y_pred))