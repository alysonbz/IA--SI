import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


# Carregar dataset
drug_data = pd.read_csv("drug200.csv")

# Separar features e rótulos
X = drug_data.drop(columns=['Drug'])  # Features
y = drug_data['Drug']  # Rótulos

# Identificar colunas categóricas
categorical_columns = X.select_dtypes(include=['object']).columns.tolist()

# Criar um transformador para codificar colunas categóricas e normalizar colunas numéricas
preprocessor = ColumnTransformer([
    ('onehot', OneHotEncoder(handle_unknown='ignore'), categorical_columns),  # Codifica as colunas categóricas
    ('scaler', StandardScaler(), X.select_dtypes(include=['int64', 'float64']).columns.tolist())  # Normaliza as numéricas
])

# Aplicar transformação nos dados
X_transformed = preprocessor.fit_transform(X)

# Dividir dataset em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.3, random_state=42)

# Definir parâmetros para busca
param_grid = {
    'n_neighbors': list(range(1, 11)),  # Testa valores ímpares de k de 1 a 19
    'metric': ['euclidean', 'manhattan', 'chebyshev', 'minkowski']
}

# Criar e executar GridSearchCV
knn = KNeighborsClassifier()
grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Exibir melhores parâmetros e melhor acurácia
print(f"Melhores parâmetros: {grid_search.best_params_}")
print(f"Melhor acurácia no treino: {grid_search.best_score_:.2f}")

# Avaliação no conjunto de teste
best_knn = grid_search.best_estimator_
test_accuracy = best_knn.score(X_test, y_test)
print(f"Acurácia no conjunto de teste: {test_accuracy:.2f}")