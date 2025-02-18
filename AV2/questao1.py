import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("C:/Users/aryel/Downloads/archive/archive/Cancer_Data.csv")

df_atualizado = df.drop(columns=['Unnamed: 32', 'id'])
df_atualizado['diagnosis'] = df_atualizado['diagnosis'].map({'M': 1, 'B': 0})
df_atualizado = df_atualizado.replace([np.inf, -np.inf], np.nan).dropna()

X = df_atualizado.drop(columns=['diagnosis'])
y = df_atualizado['diagnosis']

scaler = StandardScaler()
X = scaler.fit_transform(X)
X = np.nan_to_num(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

param_grid = {
    'n_neighbors': [3, 5, 7, 9],
    'metric': ['euclidean', 'manhattan', 'chebyshev']
}

knn = KNeighborsClassifier()

n_cores = joblib.cpu_count()
grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy', n_jobs=n_cores)
grid_search.fit(X_train, y_train)

print("\nResultados por métrica de distância:")
results = pd.DataFrame(grid_search.cv_results_)
for metric in param_grid['metric']:
    metric_results = results[results['param_metric'] == metric]
    print(f"\nMétrica: {metric}")
    for k in param_grid['n_neighbors']:
        acc = metric_results[metric_results['param_n_neighbors'] == k]['mean_test_score'].values[0]
        print(f"  k={k}: {acc:.4f}")

print("\nMelhor combinação de parâmetros:", grid_search.best_params_)

y_pred = grid_search.best_estimator_.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("\nAcurácia do modelo com os melhores parâmetros:", accuracy)

print("\nMatriz de confusão:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Benigno', 'Maligno'], yticklabels=['Benigno', 'Maligno'])
plt.xlabel('Previsto')
plt.ylabel('Real')
plt.title('Matriz de Confusão')
plt.show()
