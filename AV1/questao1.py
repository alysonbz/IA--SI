import pandas as pd

#Onde está o dataset
water_path = r'C:\Users\Bill0ca\PycharmProjects\IA--SI\AV1\Datasets\waterQuality1.csv'
water = pd.read_csv(water_path)

#Exclusão das celulas vazias e criação de uma nova caso tenha valores NaN
#Indentificação dos valores #NUM!
invalid_rows = water[water['is_safe'] == '#NUM!']
print("\nEntradas invalidas por valores #NUM!\n", invalid_rows)

#Tratamento dos valores #NUM! para NaN
water['is_safe'] = pd.to_numeric(water['is_safe'], errors='coerce')

if water.isnull().sum().sum() > 0:
    print("Valores NaN detectados. Novo arquivo criado!")
    water_cleaned = water.dropna()
else:
    print("Sem valores NaN")
    water_cleaned = water.copy()

#Colunas relevantes
selected_colums = [
    'aluminium', 'ammonia', 'arsenic', 'barium', 'cadmium', 'chloramine',
    'chromium', 'copper', 'flouride', 'bacteria', 'viruses', 'lead',
    'nitrates', 'nitrites', 'mercury', 'perchlorate', 'radium', 'selenium',
    'silver', 'uranium', 'is_safe'
]
water_relevant = water_cleaned[selected_colums]

#Distribuição de classes
print("\nDistribuição de classes:")
print(water_relevant['is_safe'].value_counts())

#Dataframe atual
print("\nDataframe atual:\n", water_relevant.head())

water_relevant.to_csv("waterQuality_ajustado.csv", index=False)
print("\nDataset ajustado salvo como 'waterQuality_ajustado.csv'.")