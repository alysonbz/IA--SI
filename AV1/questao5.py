import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder


# Importando o dataset
regressao = pd.read_csv('./dataset/data_regressao.csv')

# Exploração inicial do dataset
print('Exploração do dataset')
print(regressao.head(10))
print('\nVerificação do número de linhas e colunas:', regressao.shape)
print('\nInformações do dataset:')
print(regressao.info())
print('\nEstatísticas do dataset:')
print(regressao.describe().T)

# Verificação de valores nulos e duplicatas
print('\nVerificação de valores nulos:')
print(regressao.isna().sum())
print('\nVerificação de duplicatas:')
print(regressao.duplicated().sum())

# Verificação de valores categóricos
print('\nValores únicos por categoria:')
print('Sex:', regressao['sex'].value_counts())
print('Smoker:', regressao['smoker'].value_counts())
print('Region:', regressao['region'].value_counts())

# Visualizações de distribuição
sns.countplot(x='sex', data=regressao)
plt.title('Distribuição por Sexo')
plt.show()

sns.countplot(x='smoker', data=regressao)
plt.title('Distribuição por Status de Fumante')
plt.show()

sns.countplot(x='region', data=regressao)
plt.title('Distribuição por Região')
plt.show()

# Transformar colunas categóricas em numéricas
label_encoder = LabelEncoder()
regressao['sex'] = label_encoder.fit_transform(regressao['sex'])
regressao['smoker'] = label_encoder.fit_transform(regressao['smoker'])
regressao['region'] = label_encoder.fit_transform(regressao['region'])

# Análise gráfica de variáveis relevantes
sns.scatterplot(x='bmi', y='charges', data=regressao)
plt.title('Relação entre BMI e Charges')
plt.show()

sns.scatterplot(x='age', y='charges', data=regressao)
plt.title('Relação entre Idade e Charges')
plt.show()

sns.boxplot(x='smoker', y='charges', data=regressao)
plt.title('Impacto do Status de Fumante em Charges')
plt.show()

# Salvar o Dataset atualizado:
regressao.to_csv('./dataset_regressao_ajustado.csv', index=False)