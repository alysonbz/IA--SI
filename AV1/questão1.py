import pandas as pd

# 1. Carregar o dataset
df = pd.read_csv('/home/userpet/Downloads/gender_classification_v7.xls')
print(df)

# 2. Tratar valores ausentes (NaN)
# Remover todas as linhas com valores ausentes
df_cleaned = df.dropna()
print(df_cleaned)

# 3. Seleção das colunas mais relevantes
# Vamos verificar as primeiras linhas e as colunas para entender a estrutura dos dados
print(df_cleaned.head())
colunas_relevantes = ['long_hair', 'forehead_width_cm', 'gender']
df_relevantes = df_cleaned[colunas_relevantes]
print(df_relevantes)

# 4. Verificar a distribuição das classes (Gênero)
print(df_relevantes['gender'].value_counts())

# 5. Renomear a coluna de classes para valores numéricos
# Caso a coluna 'Gender' tenha valores categóricos como 'Male' e 'Female', converta para numéricos
df_relevantes['gender'] = df_relevantes['gender'].map({'Male': 0, 'Female': 1})
print ("\depois de trocar os nomes Male and Female for 1 or 0:")
print (df_relevantes)

# 6. Outros pré-processamentos necessários
# não fiz nenhum pre-processamento a mais

#7 salvar o data set atualizado
df_relevantes.to_csv('gender_classification_ajustado.csv', index=False)
