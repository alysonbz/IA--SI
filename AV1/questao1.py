import pandas as pd

# 1. Importação do conjunto de dados e bibliotecas
# Substituímos a URL por um caminho local para o arquivo CSV baixado manualmente
caminho_arquivo = r"C:\Users\Administrator\Downloads\healthcare-dataset-stroke-data.csv"
try:
    dados = pd.read_csv(caminho_arquivo)
except FileNotFoundError:
    print("Erro: O arquivo CSV não foi encontrado. Certifique-se de que o caminho está correto.")
    exit()
except Exception as e:
    print(f"Erro ao carregar o conjunto de dados: {e}")
    exit()

# 2. Verificação de valores nulos ou NaN e criação de um novo DataFrame sem eles
if dados.isnull().sum().any():
    dados_limpos = dados.dropna()
else:
    dados_limpos = dados.copy()

# 3. Análise das colunas mais relevantes
# Aqui, as colunas mais importantes podem ser selecionadas baseando-se em um entendimento inicial do problema.
# Exemplo de seleção de colunas (ajuste conforme necessário):
colunas_relevantes = ['gender', 'age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi', 'smoking_status', 'stroke']
try:
    dados_relevantes = dados_limpos[colunas_relevantes]
except KeyError as e:
    print(f"Erro: Algumas colunas esperadas não estão no conjunto de dados. Detalhes: {e}")
    exit()

# 4. Exibição do DataFrame final e da distribuição de classes
def distribuicao_classes(dados, coluna):
    distribuicao = dados[coluna].value_counts(normalize=True) * 100
    print("\nDistribuição de classes (em %):")
    print(distribuicao)

print("\nDataFrame Final:")
print(dados_relevantes.head())
distribuicao_classes(dados_relevantes, 'stroke')

# 5. Renomeação de valores na coluna de classes (se necessário)
# Exemplo: converter 'stroke' de categórico para valores numéricos
if dados_relevantes['stroke'].dtype == 'object':
    dados_relevantes['stroke'] = dados_relevantes['stroke'].map({"yes": 1, "no": 0})

# 6. Avaliação de necessidade de mais pré-processamento
# Exemplo: Transformação de variáveis categóricas em numéricas com one-hot encoding
colunas_categoricas = ['gender', 'smoking_status']
dados_preprocessados = pd.get_dummies(dados_relevantes, columns=colunas_categoricas, drop_first=True)

# 7. Salvando o conjunto de dados ajustado
caminho_saida = "conjunto_dados_ajustado_para_predicao.csv"
dados_preprocessados.to_csv(caminho_saida, index=False)

print(f"\nConjunto de dados ajustado salvo como: {caminho_saida}")
