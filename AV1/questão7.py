import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import KFold

# Carregar o dataset
df = pd.read_csv('/home/kali/Downloads/kc_house_data.csv')

X = df[['sqft_living']]
y = df['price']

k = 5
kf = KFold(n_splits=k, shuffle=True, random_state=42)

# Inicia lista para armazenar métricas
models = {'Linear': LinearRegression(), 'Ridge': Ridge(alpha=1.0), 'Lasso': Lasso(alpha=0.1)}
metrics = {name: {'RSS': [], 'MSE': [], 'RMSE': [], 'R2': []} for name in models}


# Função para calcular métricas
def calculate_metrics(y_true, y_pred):
    rss = np.sum((y_true - y_pred) ** 2)
    mse = rss / len(y_true)
    rmse = np.sqrt(mse)
    tss = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (rss / tss)
    return rss, mse, rmse, r2


# k-fold
for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        rss, mse, rmse, r2 = calculate_metrics(y_test, y_pred)

        metrics[name]['RSS'].append(rss)
        metrics[name]['MSE'].append(mse)
        metrics[name]['RMSE'].append(rmse)
        metrics[name]['R2'].append(r2)

# Analisar os resultados médios das métricas
for name, metric_values in metrics.items():
    print(f"\nModelo: {name}")
    for metric_name, values in metric_values.items():
        print(f"{metric_name}: {np.mean(values):.2f}")
