import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler

def load_activision_blizzard():
    return pd.read_csv('activision_blizzard.csv')

activision = load_activision_blizzard()

#Colunas relevantes
X = activision[['High']].values  #Preditor
y = activision['Close'].values  #Alvo

#K-fold Manual
def k_fold_split(X, y, k):
    fold_size = len(X) // k
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    folds = []

    for i in range(k):
        start = i * fold_size
        end = start + fold_size if i < k - 1 else len(X)
        fold_indices = indices[start:end]
        folds.append(fold_indices)

    return folds

#Métricas
def calculate_metrics(y_true, y_pred):
    rss = np.sum((y_true - y_pred) ** 2)  # Residual Sum of Squares
    mse = np.mean((y_true - y_pred) ** 2)  # Mean Squared Error
    rmse = np.sqrt(mse)  # Root Mean Squared Error
    r_squared = 1 - (rss / np.sum((y_true - np.mean(y_true)) ** 2))  # R^2
    return rss, mse, rmse, r_squared

#K-Fold Cross-Validation
k = 5
folds = k_fold_split(X, y, k)

# Modelos
results = {'Linear Regression': [], 'Ridge': [], 'Lasso': []}

for fold in folds:
    #Divisão dos dados em treino e teste
    X_train = np.delete(X, fold, axis=0)
    y_train = np.delete(y, fold, axis=0)
    X_test = X[fold]
    y_test = y[fold]

    #Normalização
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    #Regressão Linear
    model_lr = LinearRegression()
    model_lr.fit(X_train_scaled, y_train)
    y_pred_lr = model_lr.predict(X_test_scaled)
    results['Linear Regression'].append(calculate_metrics(y_test, y_pred_lr))

    #Ridge
    model_ridge = Ridge(alpha=1.0)
    model_ridge.fit(X_train_scaled, y_train)
    y_pred_ridge = model_ridge.predict(X_test_scaled)
    results['Ridge'].append(calculate_metrics(y_test, y_pred_ridge))

    #Lasso
    model_lasso = Lasso(alpha=0.1)
    model_lasso.fit(X_train_scaled, y_train)
    y_pred_lasso = model_lasso.predict(X_test_scaled)
    results['Lasso'].append(calculate_metrics(y_test, y_pred_lasso))

metrics_summary = {model: np.mean(np.array(results[model]), axis=0) for model in results}

#Resultados
best_model = None
best_metric = float('inf')  # Inicializa com infinito para RSS
best_r_squared = -float('inf')  # Inicializa com -infinito para R^2
best_metrics = None  # Para armazenar as métricas do melhor modelo

#Print das metricas
for model in metrics_summary:
    print(f"{model} Metrics:")
    print(f"RSS: {metrics_summary[model][0]:.8f}")
    print(f"MSE: {metrics_summary[model][1]:.8f}")
    print(f"RMSE: {metrics_summary[model][2]:.8f}")
    print(f"R^2: {metrics_summary[model][3]:.8f}")
    print()

    #Verificar o melhor modelo
    if metrics_summary[model][0] < best_metric:  # Melhor RSS
        best_metric = metrics_summary[model][0]
        best_model = model
        best_metrics = metrics_summary[model]  # Armazena as métricas do melhor modelo
    if metrics_summary[model][3] > best_r_squared:  # Melhor R^2
        best_r_squared = metrics_summary[model][3]

#Print do melhor modelo e métricas
print(f"O melhor modelo com base nas métricas é: {best_model}")
print(f"RSS: {best_metrics[0]:.8f}")
print(f"MSE: {best_metrics[1]:.8f}")
print(f"RMSE: {best_metrics[2]:.8f}")
print(f"R^2: {best_metrics[3]:.8f}")