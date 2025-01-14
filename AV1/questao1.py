import pandas as pd

# 1. puxar o dataset pelo caminnho
file_path = r"C:\Users\Administrator\Downloads\healthcare-dataset-stroke-data.csv"
try:
    data = pd.read_csv(file_path)
except FileNotFoundError:
    print("Erro: O arquivo CSV não foi encontrado. Certifique-se de que o caminho está correto.")
    exit()
except Exception as e:
    print(f"Erro ao carregar o dataset: {e}")
    exit()

# 2. Verificação de valores nulos e a criação de um novo dataset
if data.isnull().sum().any():
    data_cleaned = data.dropna()
else:
    data_cleaned = data.copy()

# 3. colunas relevantes
relevant_columns = ['gender', 'age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi', 'smoking_status', 'stroke']
try:
    data_relevant = data_cleaned[relevant_columns]
except KeyError as e:
    print(f"Erro: Algumas colunas esperadas não estão no dataset. Detalhes: {e}")
    exit()

# 4. dataset final e distribuição
def class_distribution(data, column):
    distribution = data[column].value_counts(normalize=True) * 100
    print("\nDistribuição de classes (em %):")
    print(distribution)

print("\nDataFrame Final:")
print(data_relevant.head())
class_distribution(data_relevant, 'stroke')

# 5. Renomeação de valores na coluna de classes (se necessário)
if data_relevant['stroke'].dtype == 'object':
    data_relevant['stroke'] = data_relevant['stroke'].map({"yes": 1, "no": 0})


categorical_columns = ['gender', 'smoking_status']
data_preprocessed = pd.get_dummies(data_relevant, columns=categorical_columns, drop_first=True)

# 7. dataset ajustavo e seu salvamento com o novo nome 
output_path = "stroke_prediction_dataset_ajustado.csv"
data_preprocessed.to_csv(output_path, index=False)

print(f"\nDataset ajustado salvo como: {output_path}")