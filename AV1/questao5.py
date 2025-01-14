import pandas as pd
import matplotlib.pyplot as plt

datasetO = pd.read_csv(r'C:\Users\adrie\OneDrive\Documentos\IA-SI_2\IA--SI\AV1\possum.csv')

#Informações do dataset
print(datasetO.info)

#Verificação de valores nulos (age, footlgth)
print(f'Valores nulos: {datasetO.isnull().sum()}')

#Excluir linhas com valores null
no_null_dataset = datasetO.dropna(subset=['age', 'footlgth'])

#Informações do dataset após exluir linhas com valores null
print(no_null_dataset.info)

#Análise colunas relevantes
print(no_null_dataset.max(), no_null_dataset.min())

new_dataset = no_null_dataset[['sex', 'age', 'hdlngth', 'skullw', 'taill', 'footlgth', 'earconch', 'belly', 'eye']]

dataset = pd.read_csv(r'C:\Users\adrie\OneDrive\Documentos\IA-SI_2\IA--SI\AV1\possum_ajustado.csv')

print(dataset.head())
print(dataset.max()) #Age tem diferença de 88,8%
print(dataset.min())

X = dataset.drop('age', axis=1).values
y = dataset['age'].values
print(type(X), type(y))

x_bm1 = X[:,3] #Rabo
print(y.shape, x_bm1.shape)

x_bm1 = x_bm1.reshape(-1, 1)
print(x_bm1.shape)

plt.scatter(x_bm1, y)
plt.ylabel('Age')
plt.xlabel('Taill')
plt.show()

x_bm2 = X[:,1] #
print(y.shape, x_bm1.shape)

x_bm2 = x_bm2.reshape(-1, 1) #Comprimento
print(x_bm1.shape)

plt.scatter(x_bm2, y)
plt.ylabel('age')
plt.xlabel('hdjngth')
plt.show()

x_bm3 = X[:,6] #
print(y.shape, x_bm3.shape)

x_bm3 = x_bm3.reshape(-1, 1) #Barriga
print(x_bm1.shape)

plt.scatter(x_bm3, y)
plt.ylabel('Age')
plt.xlabel('Belly')
plt.show()