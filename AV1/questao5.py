import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

def load_activision_blizzard():
    return pd.read_csv('activision_blizzard.csv')

activision = load_activision_blizzard()

#Verificar valores NaN
print(activision.isnull().sum())

#Remover linhas com NaN
activision_cleaned = activision.dropna()

#Colunas relevantes
activision_relevant = activision_cleaned[['Open', 'High', 'Low', 'Close', 'Adj Close']]

#Features e Target
X = activision_relevant[['Open', 'High', 'Low', 'Adj Close']]
y = activision_relevant['Close']

#Divisão o conjunto de dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#Regressão linear
model = LinearRegression()
model.fit(X_train, y_train)

#Obter os coeficientes
coefficients = model.coef_
feature_names = X.columns

#Criar um DataFrame para visualizar os coeficientes
coef_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefficients})
coef_df['Absolute Coefficient'] = coef_df['Coefficient'].abs()
coef_df = coef_df.sort_values(by='Absolute Coefficient', ascending=False)

#Exibir os coeficientes
print("\nCoeficientes da Regressão Linear:")
print(coef_df)

#Identificar o atributo mais relevante
most_relevant_attribute = coef_df['Feature'].iloc[0]
print(f'\nO atributo mais relevante para prever Close é: {most_relevant_attribute}')

#Gráfico de barras para os coeficientes
plt.figure(figsize=(10, 6))
sns.barplot(x='Coefficient', y='Feature', data=coef_df)
plt.title('Coeficientes da Regressão Linear')
plt.xlabel('Coeficiente')
plt.ylabel('Atributo')
plt.grid()
plt.axvline(0, color='red', linestyle='--')
plt.show()

#Gráfico de regressão para o atributo mais relevante
plt.figure(figsize=(10, 6))
sns.regplot(x=activision_relevant[most_relevant_attribute], y=activision_relevant['Close'], scatter_kws={'color': 'blue', 'alpha': 0.5}, line_kws={'color': 'red'})
plt.title(f'Gráfico de Regressão: Relação entre {most_relevant_attribute} e Close')
plt.xlabel(most_relevant_attribute)
plt.ylabel('Close')
plt.grid()
plt.show()