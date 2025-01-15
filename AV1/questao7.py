from src.utills import houses_sales_processed_dataset
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

houses = houses_sales_processed_dataset()

X = houses[['sqft_living']].values  # Substitua por 'sqft_living'
y = houses['price'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

k = 5
kf = KFold(n_splits=k, shuffle=True, random_state=42)

models = {
    "Linear Regression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=0.1)
}

results = {model_name: {"RSS": [], "MSE": [], "RMSE": [], "R^2": []} for model_name in models}

def calculate_metrics(y_true, y_pred):
    rss = np.sum((y_true - y_pred) ** 2)
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    r_squared = 1 - (rss / np.sum((y_true - np.mean(y_true)) ** 2))
    return rss, mse, rmse, r_squared

for train_index, test_index in kf.split(X_scaled):
    X_train, X_test = X_scaled[train_index], X_scaled[test_index]
    y_train, y_test = y[train_index], y[test_index]

    for model_name, model in models.items():
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        rss, mse, rmse, r_squared = calculate_metrics(y_test, y_pred)

        results[model_name]["RSS"].append(rss)
        results[model_name]["MSE"].append(mse)
        results[model_name]["RMSE"].append(rmse)
        results[model_name]["R^2"].append(r_squared)

final_results = {}
for model_name, metrics in results.items():
    final_results[model_name] = {
        metric: np.mean(values) for metric, values in metrics.items()
    }

print("Resultados Finais (médias das métricas para cada modelo):")
for model_name, metrics in final_results.items():
    print(f"\n{model_name}:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")