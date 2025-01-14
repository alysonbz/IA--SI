import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold


file_path = r"C:\Users\vitor\Downloads\IA.BLACK\IA--SI\AV1\Ferrari (20.04.23 - 01.05.24).csv"
df = pd.read_csv(file_path)


target = "Close"
X, y = df.drop(columns=[target]).values, df[target].values


models = {
    "Linear Regression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=0.1),
}


def calculate_metrics(y_true, y_pred):
    return {
        "RSS": np.sum((y_true - y_pred) ** 2),
        "MSE": mean_squared_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R²": r2_score(y_true, y_pred),
    }


kf = KFold(n_splits=5, shuffle=True, random_state=42)
results = {name: [] for name in models}

for train_idx, test_idx in kf.split(X):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results[name].append(calculate_metrics(y_test, y_pred))


avg_results = {
    name: {metric: np.mean([fold[metric] for fold in metrics]) for metric in metrics[0]}
    for name, metrics in results.items()
}


print("Média das Métricas por Modelo (k-fold):")
for name, metrics in avg_results.items():
    print(f"\n{name}:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")

best_model = max(avg_results, key=lambda name: avg_results[name]["R²"])
print(f"\nO melhor modelo é: {best_model} com R² médio de {avg_results[best_model]['R²']:.4f}")