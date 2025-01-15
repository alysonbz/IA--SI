import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Carregar o dataset

df = pd.read_csv('/home/kali/Downloads/kc_house_data.csv')

# Exibir as primeiras linhas do dataset
print(df.head())

# Verificar valores ausentesSS
print(df.isnull().sum())

# Remover colunas insignificantes
df = df.drop(columns=['id', 'date'])

# Calcular a correlação com o preço de venda
correlation = df.corr()
print(correlation['price'].sort_values(ascending=False))

# Visualizar a correlação das principais variáveis com o preço
plt.figure(figsize=(10, 8))
sns.heatmap(correlation[['price']].sort_values(by='price', ascending=False), annot=True, cmap='coolwarm')
plt.title('Correlação das variáveis com o preço de venda')
plt.show()

# Scatter plot da variável mais correlacionada com o preço
plt.figure(figsize=(10, 6))
sns.scatterplot(x='sqft_living', y='price', data=df)
plt.title('Relação entre sqft_living e preço de venda')
plt.xlabel('Área habitável (sqft)')
plt.ylabel('Preço de venda')
plt.show()
