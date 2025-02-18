import pandas as pd
#import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# Carregar dataset
df = pd.read_csv(r"C:\Users\gabriel\Downloads\archive\gender_classification_v7.csv")

# Separar featur e target
X = df.drop(columns=["gender"])  # Substitua "gender" pelo nome correto da coluna alvo
y = df["gender"]

# Dividir em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# normalizar os dados
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# defini os parâmetros para o GridSearchCV
param_grid = {
    'n_neighbors': range(1, 21),  # Testando valores de k de 1 a 20
    'metric': ['euclidean', 'manhattan', 'minkowski']  # Testando diferentes métricas de distância
}
# criar o kkn e aqui fazemos a busca
grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

# melhor combinação de parâmetros
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

# aqui estamos avaliando o conjunto
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Melhores parâmetros:", best_params)
print("Acurácia no conjunto de teste:", accuracy)
print("Relatório de classificação:\n", report)
