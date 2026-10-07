import pandas as pd
from sklearn.preprocessing import StandardScaler

# ========================================
# CARREGAMENTO DOS DATASETS
# ============================================
vinhored = pd.read_csv("datasetwinw/winequality-red.csv", sep=";")
vinhowhite = pd.read_csv("datasetwinw/winequality-white.csv", sep=";")

# =========================================
# 1. DIMENSÕES, TIPOS DE DADOS E VARIÁVEL CLASSE
# ===========================================
print("Dimensões Tinto (linhas, colunas):", vinhored.shape)
print("Dimensões Branco (linhas, colunas):", vinhowhite.shape)

# Imprimir os tipos de dados
print("\nTipos de dados (Tinto):\n", vinhored.dtypes)

# .unique() mostra apenas os valores possíveis sem repetição
print("\nVariável Classe ('quality') - Valores Únicos (Tinto):", sorted(vinhored['quality'].unique()))
print("Variável Classe ('quality') - Valores Únicos (Branco):", sorted(vinhowhite['quality'].unique()))


# =====================================================
# 2. VERIFICAÇÃO E TRATAMENTO DE INCONSISTÊNCIAS
# ===================================================
# 2.1 Verificar valores nulos (ausentes)
print("nulos no tinto:", vinhored.isnull().sum().sum())
print("nulos no branco:", vinhowhite.isnull().sum().sum())
# TRATAMENTO: Remover linhas duplicadas
# (Linhas exatamente iguais causam overfitting e distorcem a contagem do KNN)
vinhored_limpo = vinhored.drop_duplicates()
vinhowhite_limpo = vinhowhite.drop_duplicates()

print(f"Dimensões Tinto após remoção de duplicados: {vinhored_limpo.shape}")
print(f"Dimensões Branco após remoção de duplicados: {vinhowhite_limpo.shape}")


# ================================================
# 3. ANÁLISE ESTATÍSTICA DAS VARIÁVEIS NUMÉRICAS
# =============================================
print("\n=== 3. ANÁLISE ESTATÍSTICA (TINTO) ===")
# .describe() gera contagem, média, desvio padrão, min, quartis e máx.
# .T transpõe a tabela para facilitar a leitura no console.
print(vinhored_limpo.describe().T[['mean', 'std', 'min', '50%', 'max']])

print("\n=== 3. ANÁLISE ESTATÍSTICA (BRANCO) ===")
print(vinhowhite_limpo.describe().T[['mean', 'std', 'min', '50%', 'max']])


# =============================================
# 4. ANÁLISE DA DISTRIBUIÇÃO DAS CLASSES
# ================================
print("\n=== 4. DISTRIBUIÇÃO DAS CLASSES ('quality') ===")
# .value_counts() conta quantas amostras existem para cada nota de qualidade
print("Contagem por classe no Tinto:")
print(vinhored_limpo['quality'].value_counts().sort_index())

print("\nContagem por classe no Branco:")
print(vinhowhite_limpo['quality'].value_counts().sort_index())


# 5 e 6. AVALIAÇÃO DE ESCALAS, TRANSFORMACÕES E PREPARAÇÃO FINAL
# ==================================================
# 5.1 Separação de atributos de entrada (X) e rótulo/classe (y)
X_red = vinhored_limpo.drop(columns=['quality'])
y_red = vinhored_limpo['quality']

X_white = vinhowhite_limpo.drop(columns=['quality'])
y_white = vinhowhite_limpo['quality']

# 5.2 Aplicação da Padronização (StandardScaler)
# Transforma as variáveis para Média = 0 e Desvio Padrão = 1 (Z-score)
scaler_red = StandardScaler()
X_red_scaled = scaler_red.fit_transform(X_red)

scaler_white = StandardScaler()
X_white_scaled = scaler_white.fit_transform(X_white)

# 5.3 Reconstrução do DataFrame final com as colunas organizadas
vinhored_preparado = pd.DataFrame(X_red_scaled, columns=X_red.columns)
vinhored_preparado['quality'] = y_red.values

vinhowhite_preparado = pd.DataFrame(X_white_scaled, columns=X_white.columns)
vinhowhite_preparado['quality'] = y_white.values

# 5.4 Salvar versões finais em ficheiros CSV para a Questão 2
vinhored_preparado.to_csv("datasetwinw/winequality_red_preparado.csv", index=False)
vinhowhite_preparado.to_csv("datasetwinw/winequality-white_preparado.csv", index=False)
