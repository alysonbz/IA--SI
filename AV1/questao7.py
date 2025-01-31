import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# 1) Carregar o dataset
file_path = 'car_price.csv'  # Substitua com o caminho do seu arquivo
df = pd.read_csv(file_path)

# 2) Selecionar as colunas de interesse
selected_columns = ['Price', 'Levy', 'Prod. year', 'Leather interior','Mileage']
df = df[selected_columns]

# 3) Substituir 'Yes' por 1 e 'No' por 0
df['Leather interior'] = df['Leather interior'].replace({'Yes': 1, 'No': 0})

# 4) Processar a coluna 'Mileage' (Kilometragem)
# Vamos remover o sufixo ' km' e converter para valor numérico
df['Mileage'] = df['Mileage'].replace({' km': ''}, regex=True)  # Remove o sufixo ' km'
# Remover linhas com valores faltantes e strings
df = df.dropna()
df = df.apply(pd.to_numeric, errors='coerce')
df.dropna(inplace=True)

# 5) Separar as variáveis independentes (X) e a dependente (y)
X = df.drop(columns=['Price'])
y = df['Price']

# 6) Normalizar os dados
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# 7) Função para calcular as métricas manualmente
def calculate_metrics(y_true, y_pred):
    # RSS - Residual Sum of Squares
    rss = np.sum((y_true - y_pred) ** 2)

    # MSE - Mean Squared Error
    mse = mean_squared_error(y_true, y_pred)

    # RMSE - Root Mean Squared Error
    rmse = np.sqrt(mse)

    # R² - Coeficiente de Determinação
    r2 = r2_score(y_true, y_pred)

    return rss, mse, rmse, r2


# 8) Função para realizar a validação cruzada k-fold
def k_fold_cross_validation(X, y, model, k=5):
    fold_size = len(X) // k
    metrics = {'RSS': [], 'MSE': [], 'RMSE': [], 'R2': []}

    for i in range(k):
        # Dividir os dados em treinamento e validação
        val_start = i * fold_size
        val_end = (i + 1) * fold_size if i < k - 1 else len(X)

        X_train = np.concatenate([X[:val_start], X[val_end:]], axis=0)
        y_train = np.concatenate([y[:val_start], y[val_end:]], axis=0)
        X_val = X[val_start:val_end]
        y_val = y[val_start:val_end]

        # Treinar o modelo
        model.fit(X_train, y_train)

        # Fazer previsões
        y_pred = model.predict(X_val)

        # Calcular as métricas
        rss, mse, rmse, r2 = calculate_metrics(y_val, y_pred)

        metrics['RSS'].append(rss)
        metrics['MSE'].append(mse)
        metrics['RMSE'].append(rmse)
        metrics['R2'].append(r2)

    # Retornar a média das métricas
    return {metric: np.mean(values) for metric, values in metrics.items()}


# 9) Testar os modelos
models = {
    'Linear Regression': LinearRegression(),
    'Ridge': Ridge(),
    'Lasso': Lasso()
}

results = {}

for model_name, model in models.items():
    print(f'Treinando e avaliando o modelo {model_name}...')
    metrics = k_fold_cross_validation(X_scaled, y, model, k=5)
    results[model_name] = metrics

# 10) Exibir os resultados das métricas para cada modelo
print("\nMétricas de Avaliação para cada modelo:")
for model_name, metrics in results.items():
    print(f"\nModelo: {model_name}")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

# 11) Análise de Desempenho - Comparar os resultados
best_model = min(results, key=lambda x: results[x]['MSE'])
print(f"\nO modelo com o melhor desempenho (menor MSE) é: {best_model}")