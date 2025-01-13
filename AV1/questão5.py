import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# 1. Importar as bibliotecas necessárias
# (importações já feitas acima)

# 2. Carregar o dataset de regressão
try:
    # Tentar carregar o dataset atualizado
    df = pd.read_csv("housesalesprediction.csv")
    print("Dataset carregado.")
except FileNotFoundError:
    # Caso o arquivo não esteja disponível, carregá-lo do Kaggle
    url = "https://www.kaggleusercontent.com/datasets/harlfoxem/housesalesprediction"
    df = pd.read_csv(url)
    print("Dataset carregado do Kaggle.")

# 3. Exibir as primeiras linhas do dataframe
print(df.head())

# 4. Identificar o atributo alvo para regressão
# Supomos que o alvo para regressão seja o preço da casa, que geralmente é o atributo alvo em dados de venda de casas
target = 'SalePrice'
print(f"Atributo alvo: {target}")

# 5. Remover colunas insignificantes e tratar valores NaN
# Verificar quais colunas contêm valores NaN e remover os registros que têm valores ausentes
df = df.dropna()  # Remover linhas que contêm NaN

# 6. Verificar as colunas mais relevantes para regressão
# Um primeiro passo é verificar o desvio padrão e a correlação do preço da casa com outras variáveis
print(df.corr())

# Selecionar as colunas que são relevantes
relevant_features = df.corr()[target].sort_values(ascending=False).drop(target)

# Visualizar a correlação com o preço (alvo)
plt.figure(figsize=(10, 6))
relevant_features.plot(kind='bar', title='Correlação dos atributos com SalePrice')
plt.xlabel('Atributo')
plt.ylabel('Correlação')
plt.xticks(rotation=45)
plt.show()

# 7. Remover colunas insignificantes (com correlação baixa)
# Remover colunas com correlação menor que um valor arbitrário, como 0.1
threshold = 0.1
insignificant_cols = relevant_features[relevant_features.abs() < threshold].index
df = df.drop(columns=insignificant_cols)

# 8. Dividir o dataset em treino e teste
X = df.drop(target, axis=1)
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 9. Print do dataframe final e análise do atributo alvo e mais relevantes
print("\nDataFrame final após remoção de NaN e colunas insignificantes:")
print(X_train.head())

# Visualizar o distribuições do alvo (Preço da casa)
plt.figure(figsize=(10, 6))
y.hist(bins=50, alpha=0.7)
plt.title("Distribuição do SalePrice")
plt.xlabel("SalePrice")
plt.ylabel("Frequência")
plt.show()