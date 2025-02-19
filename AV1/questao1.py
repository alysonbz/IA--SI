import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

#Importando o dataset:
classificacao = pd.read_csv('./dataset/data_classificacao.csv')

#Exploração do dataset:
print('Exploração do dataset')
print(classificacao.head(100))

# Verificação do numero de linhas e colunas:
print('Verificação do numero de linhas e colunas')
print(classificacao.shape)

#Todas as informações do dataset:
print('Informações do dataset')
print(classificacao.info())

#Estatísticas do dataset:
print('Estatísticas do dataset')
print(classificacao.describe().T)

#Nome das colunas:
print('Colunas do dataset')
print(classificacao.columns)

#Tipo de dados de cada coluna:
print('Tipos de dados das colunas')
print(classificacao.dtypes)

#Verificação de valores nulos:
print('Verificação de valores nulos')
print(classificacao.isna().sum())

#Verificação de duplicatas:
print('Verificação de duplicatas')
print(classificacao.duplicated().sum())

#Separação de colunas com dados categóricos:
print('Colunas com dados categoricos')
categoricos_col = [cat for cat in classificacao.columns if classificacao[cat].dtype=='O']
print('Existem {} colunas categóricas'.format(len(categoricos_col)))
print(categoricos_col)

#Separação de colunas com dados numéricos:
print('Colunas com dados numericos')
numericos_col = [num for num in classificacao.columns if classificacao[num].dtype!='O']
print('Existem {} colunas numericas'.format(len(numericos_col)))
print(numericos_col)

#Selecionar colunas relevantes
print('Colunas relevantes')
colunas_relevantes = ['type_of_meal_plan', 'room_type_reserved', 'market_segment_type', 'booking_status', 'lead_time', 'avg_price_per_room', 'no_of_special_requests', 'no_of_previous_cancellations']
classificacao = classificacao[colunas_relevantes]

#Codificação de rótulo em colunas categóricas que são: type_of_meal_plan, room_type_reserved, market_segment_type, booking_status
label_enconder=LabelEncoder()
classificacao['type_of_meal_plan']=label_enconder.fit_transform(classificacao['type_of_meal_plan'])
classificacao['room_type_reserved']=label_enconder.fit_transform(classificacao['room_type_reserved'])
classificacao['market_segment_type']=label_enconder.fit_transform(classificacao['market_segment_type'])
classificacao['booking_status']=label_enconder.fit_transform(classificacao['booking_status'])
print(classificacao)

# Mostrar a distribuição de classes
print("Distribuição de classes:")
print(classificacao['booking_status'].value_counts())

#DataFrame final
print("DataFrame final após pré-processamento:")
print(classificacao.head())

# Contar ocorrências
contagem = classificacao['booking_status'].value_counts()

# Gráfico
plt.figure(figsize=(10, 5))
plt.bar(contagem.index, contagem.values, color=['skyblue', 'salmon'])
plt.xticks([0, 1], ['Canceled', 'Not_Canceled'])
plt.xlabel('Status de Reserva')
plt.ylabel('Contagem')
plt.title('Status de Reservas')
plt.show()

# Verificar necessidade de mais pré-processamento
print("O pré-processamento atual cobre todas as etapas necessárias para este exercício.")

# Salvar o dataset atualizado
classificacao.to_csv('dataset_classificacao_ajustado.csv', index=False)