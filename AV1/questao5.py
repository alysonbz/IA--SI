import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Carregar o arquivo da Acer
file_path = r'C:\Users\Administrator\Downloads\DataSet\Acer (2000-2024).csv'
df = pd.read_csv(file_path)

# Converter a coluna Date para formato de data
df['Date'] = pd.to_datetime(df['Date'])

# Selecionar apenas colunas numéricas
df_numerico = df.select_dtypes(include=['float64', 'int64'])

# Calcular a correlação com a coluna 'Close'
correlacoes = df_numerico.corr()['Close'].sort_values(ascending=False)
print("Correlação com o Preço de Fechamento (Close):\n", correlacoes)

# Plotar gráfico de calor da matriz de correlação
plt.figure(figsize=(8, 6))
sns.heatmap(df_numerico.corr(), annot=True, cmap="coolwarm")
plt.title("Matriz de Correlação - Acer")
plt.show()



