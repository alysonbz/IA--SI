import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cityblock, chebyshev, euclidean
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from collections import Counter

dataset = pd.read_csv('Dataset_coletado.csv')

X = dataset.drop(columns=['blue'])
y = dataset['blue']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

def knn(X_train, y_train, X_test, k, distance_func):
    predictions = []
    for test_point in X_test:
        distances = [(distance_func(test_point, train_point), y_train[i]) for i, train_point in enumerate(X_train)]
        distances.sort(key=lambda x: x[0])
        k_nearest = [label for _, label in distances[:k]]
        predictions.append(Counter(k_nearest).most_common(1)[0][0])
    return predictions

distance_functions = {
    'Euclidean': euclidean,
    'Manhattan': cityblock,
    'Chebyshev': chebyshev
}

k_values = range(1, 21)
results_manual = {}
for name, func in distance_functions.items():
    accuracies = []
    for k in k_values:
        y_pred = knn(X_train, y_train.to_numpy(), X_test, k, func)
        accuracy = np.mean(y_pred == y_test.to_numpy())
        accuracies.append(accuracy)
    results_manual[name] = accuracies

plt.figure(figsize=(10, 6))
for name, accuracies in results_manual.items():
    plt.plot(k_values, accuracies, label=name)
plt.xlabel('Valor de k')
plt.ylabel('Acurácia')
plt.title('KNN Manual: Acurácia vs k')
plt.legend()
plt.show()

param_grid = {
    'n_neighbors': list(k_values),
    'metric': ['euclidean', 'manhattan', 'chebyshev']
}

knn_sklearn = KNeighborsClassifier()
grid_search = GridSearchCV(knn_sklearn, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

best_k = grid_search.best_params_['n_neighbors']
best_metric = grid_search.best_params_['metric']
best_score = grid_search.best_score_

print(f"Melhor k encontrado pelo GridSearchCV: {best_k}")
print(f"Melhor métrica encontrada pelo GridSearchCV: {best_metric}")
print(f"Melhor acurácia no treino: {best_score:.2f}")

print("\nComparação de Resultados:")
for name in results_manual:
    best_k_manual = k_values[np.argmax(results_manual[name])]
    best_acc_manual = max(results_manual[name])
    print(f"{name}: Melhor k (Manual) = {best_k_manual}, Melhor Acurácia = {best_acc_manual:.2f}")
    
if best_k_manual == best_k and best_metric == name.lower():
    print("Os resultados do GridSearchCV e do KNN manual são consistentes!")
else:
    print("Houve diferenças entre os resultados do GridSearchCV e do KNN manual.")

