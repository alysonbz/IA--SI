import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score

# 1. Carregar o dataset
try:
    df = pd.read_csv("housesalesprediction.csv")
    print("Dataset carregado.")
except FileNotFoundError:
    url = "https://www.kaggleusercontent.com/datasets/harlfoxem/housesalesprediction"
    df = pd.read_csv(url)
    print("Dataset carregado do Kaggle.")

# 2. Exibir as primeiras linhas do dataframe
print(df.head())

# 3. Identificar o atributo alvo para regressão
target = 'SalePrice'

# 4. Remover colunas insignificantes e tratar valores NaN
df = df.dropna()  # Remover linhas que contêm NaN

# 5. Verificar as colunas mais relevantes para regressão
# Selecionar as colunas que são relevantes
correlation = df.corr()[target].sort_values(ascending=False).drop(target)

# Atributo mais relevante
relevant_feature = correlation.idxmax()

# 6. Dividir o dataset em X (atributo mais relevante) e y (target)
X = df[[relevant_feature]].values
y = df[target].values

# 7. K-Fold Cross Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Modelos a serem testados
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),  # Pode ajustar o valor de alpha para experimentar
    "Lasso Regression": Lasso(alpha=0.1)  # Ajustar o valor de alpha conforme necessário
}

# Resultados para armazenamento
results = {model: {'RSS': [], 'MSE': [], 'RMSE': [], 'R^2': []} for model in models.keys()}

# 8. Loop para validação cruzada
for model_name, model in models.items():
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Treinar o modelo
        model.fit(X_train, y_train)

        # Fazer previsões
        y_pred = model.predict(X_test)

        # 9. Calcular as métricas manualmente
        residuals = y_test - y_pred
        RSS = np.sum(residuals ** 2)
        MSE = mean_squared_error(y_test, y_pred)
        RMSE = np.sqrt(MSE)
        R_squared = r2_score(y_test, y_pred)

        # Armazenar os resultados
        results[model_name]['RSS'].append(RSS)
        results[model_name]['MSE'].append(MSE)
        results[model_name]['RMSE'].append(RMSE)
        results[model_name]['R^2'].append(R_squared)

# 10. Cálculo das métricas médias para cada modelo
mean_results = {}
for model_name, metrics in results.items():
    mean_results[model_name] = {
        'Mean RSS': np.mean(metrics['RSS']),
        'Mean MSE': np.mean(metrics['MSE']),
        'Mean RMSE': np.mean(metrics['RMSE']),
        'Mean R^2': np.mean(metrics['R^2'])
    }

# 11. Análise de desempenho
print("\nAnálise de Desempenho:")
for model_name, mean_metric in mean_results.items():
    print(f"\nModelo: {model_name}")
    print(f"Mean RSS: {mean_metric['Mean RSS']:.4f}")
    print(f"Mean MSE: {mean_metric['Mean MSE']:.4f}")
    print(f"Mean RMSE: {mean_metric['Mean RMSE']:.4f}")
    print(f"Mean R^2: {mean_metric['Mean R^2']:.4f}")