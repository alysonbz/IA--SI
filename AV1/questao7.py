import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# 1. Importar o dataset
regressao = pd.read_csv('./dataset/dataset_regressao_ajustado.csv')

# 2. Escolher o atributo mais relevante para o modelo
X = regressao[['smoker']]  # Variável independente
y = regressao['charges']  # Variável dependente

# 3. Normalização dos dados
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# Função para calcular as métricas manualmente
def calculate_metrics(y_true, y_pred):
    # RSS (Residual Sum of Squares)
    RSS = np.sum((y_true - y_pred) ** 2)

    # MSE (Mean Squared Error)
    MSE = mean_squared_error(y_true, y_pred)

    # RMSE (Root Mean Squared Error)
    RMSE = np.sqrt(MSE)

    # R² (Coeficiente de Determinação)
    R2 = r2_score(y_true, y_pred)

    return RSS, MSE, RMSE, R2


# Implementação do K-Fold Cross-Validation manualmente
k = 5  # Definir o número de folds
fold_size = len(X) // k

# Inicializando o armazenamento dos resultados das métricas
results = {'LinearRegression': {'RSS': [], 'MSE': [], 'RMSE': [], 'R2': []},
           'Ridge': {'RSS': [], 'MSE': [], 'RMSE': [], 'R2': []},
           'Lasso': {'RSS': [], 'MSE': [], 'RMSE': [], 'R2': []}}

for i in range(k):
    # Dividir os dados em treino e validação
    validation_start = i * fold_size
    validation_end = (i + 1) * fold_size if i != k - 1 else len(X)

    X_train = np.concatenate([X_scaled[:validation_start], X_scaled[validation_end:]], axis=0)
    X_valid = X_scaled[validation_start:validation_end]

    y_train = np.concatenate([y[:validation_start], y[validation_end:]], axis=0)
    y_valid = y[validation_start:validation_end]

    # 4. Modelos a serem testados (Regressão Linear, Ridge, Lasso)
    models = {
        'LinearRegression': LinearRegression(),
        'Ridge': Ridge(alpha=1),
        'Lasso': Lasso(alpha=1)
    }

    for model_name, model in models.items():
        # Treinar o modelo
        model.fit(X_train, y_train)

        # Fazer previsões
        y_pred = model.predict(X_valid)

        # Calcular as métricas
        RSS, MSE, RMSE, R2 = calculate_metrics(y_valid, y_pred)

        # Armazenar os resultados nas listas correspondentes
        results[model_name]['RSS'].append(RSS)
        results[model_name]['MSE'].append(MSE)
        results[model_name]['RMSE'].append(RMSE)
        results[model_name]['R2'].append(R2)

# 5. Calcular a média das métricas para cada modelo
averages = {}
for model_name, metrics in results.items():
    averages[model_name] = {metric: np.mean(values) for metric, values in metrics.items()}

# Exibir os resultados
for model_name, avg_metrics in averages.items():
    print(f"\n{model_name}:")
    for metric, avg_value in avg_metrics.items():
        print(f"Average {metric}: {avg_value}")

# 6. Comparar os modelos
best_model = min(averages, key=lambda model: averages[model]['RMSE'])  # Escolher o modelo com o menor RMSE
print(f"\nMelhor modelo: {best_model}")