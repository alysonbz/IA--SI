import pandas as pd

#Onde está o dataset
water_path = r'C:\Users\Bill0ca\PycharmProjects\IA--SI\AV1\Datasets\waterQuality1.csv'
water = pd.read_csv(water_path)

#Exclusão das celulas vazias e criação de uma nova
if water.isnull().values.any():
    print("Valores NaN detectados")
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

#Distribuição convertida
water_relevant['is_safe'] = water_relevant['is_safe'].astype('category').cat.codes
print("\nClasses convertidas para valores numéricos.")

#Dataframe atual
print("\nDataframe atual:\n", water_relevant.head())

water_relevant.to_csv("waterQuality_modificado.csv", index=False)
print("\nDataset ajustado salvo como 'waterQuality_modificado.csv'.")