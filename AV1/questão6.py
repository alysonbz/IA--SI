import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1. Importar as bibliotecas necessárias
# (importações já feitas acima)

# 2. Carregar o dataset
try:
    df = pd.read_csv("housesalesprediction.csv")
    print("Dataset carregado.")
except FileNotFoundError:
    url = "https://www.kaggleusercontent.com/datasets/harlfoxem/housesalesprediction"
    df = pd.read_csv(url)
    print("Dataset carregado do Kaggle.")

# 3. Exibir as primeiras linhas do dataframe
print(df.head())

# 4. Identificar o atributo alvo para regressão
target = 'SalePrice'

# 5. Remover colunas insignificantes e tratar valores NaN
df = df.dropna()  # Remover linhas que contêm NaN

# 6. Verificar as colunas mais relevantes para regressão
# Selecionar as colunas que são relevantes
correlation = df.corr()[target].sort_values(ascending=False).drop(target)

# Atributo mais relevante
relevant_feature = correlation.idxmax()
print(f"Atributo mais relevante para regressão: {relevant_feature}")

# Visualizar a correlação com o preço (alvo)
plt.figure(figsize=(10, 6))
correlation.plot(kind='bar', title='Correlação dos atributos com SalePrice')
plt.xlabel('Atributo')
plt.ylabel('Correlação')
plt.xticks(rotation=45)
plt.show()

# 7. Dividir o dataset em treino e teste
X = df[[relevant_feature]].values
y = df[target].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 8. Fitting the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# 9. Previsão com o modelo treinado
y_pred = model.predict(X_test)

# 10. Plot da reta de regressão junto com a nuvem de pontos
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color='blue', alpha=0.6, label='Pontos Reais')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Reta de Regressão')
plt.title("Reta de Regressão vs Nuvem de Pontos")
plt.xlabel(relevant_feature)
plt.ylabel('SalePrice')
plt.legend()
plt.show()

# 11. Calcular RSS, MSE, RMSE, R^2
# Residual Sum of Squares (RSS)
residuals = y_test - y_pred
RSS = np.sum(residuals**2)

# Mean Squared Error (MSE)
MSE = mean_squared_error(y_test, y_pred)

# Root Mean Squared Error (RMSE)
RMSE = np.sqrt(MSE)

# R-squared
R_squared = r2_score(y_test, y_pred)

# 12. Print dos resultados
print("\nResultados da Regressão Linear:")
print(f"RSS: {RSS:.4f}")
print(f"MSE: {MSE:.4f}")
print(f"RMSE: {RMSE:.4f}")
print(f"R-squared: {R_squared:.4f}")