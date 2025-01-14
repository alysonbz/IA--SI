import pandas as pd
import matplotlib.pyplot as plt
from src.utils import cancer_dataset

df = pd.read_csv(r"C:\Users\aryel\OneDrive\Documentos\IA_ary\IA--SI\AV1\dataset\Cancer_Data.csv")

print("O tamanho do dataset é", df.shape)

print(df.isnull().sum())

df_atualizado = df.drop(columns=['Unnamed: 32', 'id'])
print("Coluna 'Unnamed: 32' removida")

# Exibir o DataFrame final atualizado
print("\nDataFrame Final Atualizado:")
print(df_atualizado)

df_atualizado['diagnosis'] = df_atualizado['diagnosis'].map({'M':1, 'B':0})
print("\nDistribuição de Classes para a coluna 'diagnosis':")
print(df_atualizado['diagnosis'].value_counts())

df_atualizado.to_csv(r"C:\Users\aryel\OneDrive\Documentos\IA_ary\IA--SI\AV1\dataset\Cancer_Data_ajustado.csv")

df_atualizado['diagnosis'].value_counts().plot(kind='pie', autopct='%1.1f%%', startangle=90, colors=['lightblue', 'salmon'])
plt.title('Distribuição de Diagnóstico (Benigno vs Maligno)')
plt.ylabel('')
plt.show()

