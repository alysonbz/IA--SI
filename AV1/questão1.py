import pandas as pd

# 1. Importe as bibliotecas necessárias
# (já importado acima)

# 2. Carregue o dataset e verifique células vazias ou NaN
try:
    # Carregar o dataset
    url = "https://www.kaggleusercontent.com/datasets/elakiricoder/gender-classification-dataset/data.csv"
    df = pd.read_csv(url)
    print("Dataset carregado com sucesso!")
except Exception as e:
    print(f"Erro ao carregar o dataset: {e}")
    exit()

# Verificar células vazias ou NaN
print("\nVerificando células vazias ou NaN:")
print(df.isnull().sum())

if df.isnull().sum().any():
    # Remover linhas com valores NaN e criar novo dataframe
    df = df.dropna()
    print("\nLinhas com NaN removidas.")
else:
    print("\nNenhum valor NaN encontrado.")

# 3. Verificar as colunas mais relevantes
print("\nColunas disponíveis no dataset:")
print(df.columns)

# Exemplo: Supondo que as colunas 'height', 'weight' e 'gender' sejam relevantes
columns_needed = ['height', 'weight', 'gender']
df = df[columns_needed]
print("\nNovo dataframe com colunas relevantes:")
print(df.head())

# 4. Mostrar a distribuição de classes
print("\nDistribuição das classes (coluna 'gender'):")
print(df['gender'].value_counts())

# 5. Renomear a coluna de classes para atributos numéricos, se necessário
if df['gender'].dtype == 'object':
    df['gender'] = df['gender'].map({'Male': 0, 'Female': 1})
    print("\nColuna 'gender' convertida para valores numéricos.")

# 6. Verificar necessidade de pré-processamento adicional
# Exemplo: Verificar outliers ou escala de valores
print("\nDescrição estatística do dataset:")
print(df.describe())

# (Análise visual ou lógica aqui, caso necessário)

# 7. Salvar o dataset atualizado
output_file = "gender_classification_adjusted.csv"
df.to_csv(output_file, index=False)
print(f"\nDataset atualizado salvo como: {output_file}")