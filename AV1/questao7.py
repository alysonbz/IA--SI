import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

dataset_path = 'bodyfat.csv'
df = pd.read_csv(dataset_path)

df = df.dropna()

X = df.drop("BodyFat", axis=1)
y = df["BodyFat"]

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

def k_fold_cross_validation(X, y, k, model_class, **model_kwargs):
    fold_size = len(X) // k
    results = {"RSS": [], "MSE": [], "RMSE": [], "R2": []}

    for i in range(k):
        # Divisão dos dados em treino e teste
        test_start = i * fold_size
        test_end = (i + 1) * fold_size if i < k - 1 else len(X)
        
        X_train = np.concatenate([X[:test_start], X[test_end:]])
        y_train = np.concatenate([y[:test_start], y[test_end:]])
        X_test, y_test = X[test_start:test_end], y[test_start:test_end]

        model = model_class(**model_kwargs)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        rss = np.sum((y_test - y_pred) ** 2)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        results["RSS"].append(rss)
        results["MSE"].append(mse)
        results["RMSE"].append(rmse)
        results["R2"].append(r2)

    metrics = {key: np.mean(value) for key, value in results.items()}
    return metrics

k = 5 

linear_metrics = k_fold_cross_validation(X_scaled, y, k, LinearRegression)

ridge_metrics = k_fold_cross_validation(X_scaled, y, k, Ridge, alpha=1.0)

lasso_metrics = k_fold_cross_validation(X_scaled, y, k, Lasso, alpha=0.1)

print(f"Modelo Linear (Clássico) - Métricas: {linear_metrics}")
print(f"Modelo Ridge - Métricas: {ridge_metrics}")
print(f"Modelo Lasso - Métricas: {lasso_metrics}")
models = ["Linear", "Ridge", "Lasso"]
metrics_dict = {"Linear": linear_metrics, "Ridge": ridge_metrics, "Lasso": lasso_metrics}

best_model = min(models, key=lambda model: metrics_dict[model]["RMSE"])
print(f"\nO melhor modelo é: {best_model}")