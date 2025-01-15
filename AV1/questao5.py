from src.utills import kc_house_dataset
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

kc_house = kc_house_dataset()

print("Informações gerais do dataset:")
print(kc_house.info())

print("\nValores ausentes por coluna:")
print(kc_house.isnull().sum())

kc_house_cleaned = kc_house.dropna()
print(f"\nApós remoção de valores NaN, o dataset possui {kc_house_cleaned.shape[0]} linhas.")

target = 'price'

irrelevant_columns = ['id', 'date']
houses_cleaned = kc_house_cleaned.drop(columns=irrelevant_columns)

correlation_matrix = houses_cleaned.corr()
correlation_with_target = correlation_matrix[target].sort_values(ascending=False)

print("\nCorrelação dos atributos com o alvo:")
print(correlation_with_target)

plt.figure(figsize=(10, 6))
sns.barplot(x=correlation_with_target.index, y=correlation_with_target.values)
plt.title('Correlação dos atributos com o preço')
plt.xticks(rotation=90)
plt.ylabel('Correlação')
plt.grid()
plt.show()

relevant_features = correlation_with_target[abs(correlation_with_target) > 0.5].index
print("\nAtributos mais relevantes para o alvo:")
print(relevant_features)

houses_relevant = houses_cleaned[relevant_features]

print("\nDataset processado (com atributos relevantes):")
print(houses_relevant.head())

output_path = "houses_sales_processed.csv"
houses_relevant.to_csv(output_path, index=False)
print(f"\nDataset ajustado salvo como {output_path}.")
