import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Carregando o dataset
from src.utils import load_laptopPrice_dataset
laptop_price = load_laptopPrice_dataset()

# Exibindo informações sobre o dataset
print('Corpo\n', laptop_price.shape)
print('Primeiros valores\n', laptop_price.head())
print('Informações\n', laptop_price.info())
print('Valores máximos das colunas\n', laptop_price.max())
print('Valores Mínimos das colinas\n', laptop_price.min())
print('Verificando os NaN\n', laptop_price.isnull().sum())

# Pré-processamento: Convertendo as colunas para tipos numéricos onde necessário


# Removendo a coluna 'weight' que possui muitos NaN
laptop_price = laptop_price.drop(columns=['weight'])

# Selecionando apenas as colunas numéricas
numerical_data = laptop_price.select_dtypes(include=['float64', 'int64'])

# Calculando a matriz de correlação
correlation_matrix = numerical_data.corr()

# Plotando o gráfico de correlação
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title("Correlação entre as variáveis numéricas")
plt.show()

# Verificando a correlação com o preço
print("Correlação entre as variáveis numéricas:\n", correlation_matrix)
