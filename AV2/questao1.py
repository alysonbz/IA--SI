import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from AV2.src.utills import diabetes_dataset

diabetes = diabetes_dataset()

diabetes.dropna(inplace=True)

X = diabetes.drop("Outcome", axis=1)
y = diabetes["Outcome"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

param_grid = {
    "n_neighbors": list(range(1, 21)),
    "metric": ["euclidean", "manhattan", "chebyshev", "minkowski"]
}

knn = KNeighborsClassifier()
grid_search = GridSearchCV(knn, param_grid, cv=5, scoring="accuracy", n_jobs=-1)

grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
best_score = grid_search.best_score_

print("Melhores Hiperparâmetros:", best_params)
print("Melhor Acurácia (validação cruzada):", best_score)

best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

test_accuracy = accuracy_score(y_test, y_pred)
print("\nAcurácia no Conjunto de Teste:", test_accuracy)
print("\nRelatório de Classificação no Conjunto de Teste:")
print(classification_report(y_test, y_pred))