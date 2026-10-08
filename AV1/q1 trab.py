import pandas as pd
from sklearn.preprocessing import StandardScaler

vinhored = pd.read_csv("datasetwinw/winequality-red.csv", sep=";")
vinhowhite = pd.read_csv("datasetwinw/winequality-white.csv", sep=";")

# 1. DIMENSÕES, TIPOS DE DADOS E VARIÁVEL CLASSE
print("Dimensões Tinto (linhas, colunas):", vinhored.shape)
print("Dimensões Branco (linhas, colunas):", vinhowhite.shape)

print("\nTipos de dados (Tinto):\n", vinhored.dtypes)
print("\nTipos de dados (Branco):\n", vinhowhite.dtypes)

print("Variável Classe ('quality') - Valores Únicos (Tinto):", sorted(vinhored['quality'].unique()))
print("Variável Classe ('quality') - Valores Únicos (Branco):", sorted(vinhowhite['quality'].unique()))

# 2. VERIFICAÇÃO E TRATAMENTO DE INCONSISTÊNCIAS
print("\nINCONSISTÊNCIAS")
print("Total de nulos no Tinto:", vinhored.isnull().sum().sum())
print("Total de nulos no Branco:", vinhowhite.isnull().sum().sum())

print("Linhas duplicadas no Tinto:", vinhored.duplicated().sum())
print("Linhas duplicadas no Branco:", vinhowhite.duplicated().sum())

# Checagens adicionais de domínio
negativos_red = (vinhored.drop(columns=['quality']) < 0).sum().sum()
ph_invalido_red = ((vinhored['pH'] < 0) | (vinhored['pH'] > 14)).sum()
print(f"Valores negativos (Tinto): {negativos_red} | pH fora de [0, 14] (Tinto): {ph_invalido_red}")

# Tratamento: Remoção de duplicados
vinhored_limpo = vinhored.drop_duplicates()
vinhowhite_limpo = vinhowhite.drop_duplicates()

# 3. DIMENSÕES ANTES E DEPOIS
print(f"\nTinto -> Antes: {vinhored.shape} | Depois: {vinhored_limpo.shape} | Removidos: {vinhored.shape[0] - vinhored_limpo.shape[0]} duplicados")
print(f"Branco -> Antes: {vinhowhite.shape} | Depois: {vinhowhite_limpo.shape} | Removidos: {vinhowhite.shape[0] - vinhowhite_limpo.shape[0]} duplicados")

# 4. ANÁLISE ESTATÍSTICA E EVIDÊNCIA DE ESCALAS
print("\nANÁLISE ESTATÍSTICA (TINTO LIMPO)")
print(vinhored_limpo.describe().T[['mean', 'std', 'min', '50%', 'max']])

# Tabela explícita de Amplitudes para a justificativa de escala
atributos_red = vinhored_limpo.drop(columns=['quality'])
evidencia_escala = pd.DataFrame({
    'Minimo': atributos_red.min(),
    'Maximo': atributos_red.max(),
    'Amplitude (Max - Min)': atributos_red.max() - atributos_red.min()
}).round(3)
print("EVIDÊNCIA DE AMPLITUDES (TINTO) evidencia_escala")

# 5. ANÁLISE DA DISTRIBUIÇÃO DAS CLASSES (COM %)
print("DISTRIBUIÇÃO DAS CLASSES (TINTO)")
dist_red = pd.DataFrame({
    'Contagem': vinhored_limpo['quality'].value_counts().sort_index(),
    'Percentual (%)': (vinhored_limpo['quality'].value_counts(normalize=True).sort_index() * 100).round(2)
})
print(dist_red)

# 6. EXPORTAÇÃO DOS DATASETS
# Versão 1: Escala original sem duplicados
vinhored_limpo.to_csv("datasetwinw/winequality_red_limpo.csv", index=False)
vinhowhite_limpo.to_csv("datasetwinw/winequality_white_limpo.csv", index=False)
#vinho tinto
X_red = vinhored_limpo.drop(columns=['quality'])
y_red = vinhored_limpo['quality']
scaler_red = StandardScaler()
X_red_scaled = scaler_red.fit_transform(X_red)

vinhored_preparado = pd.DataFrame(X_red_scaled, columns=X_red.columns)
vinhored_preparado['quality'] = y_red.values
vinhored_preparado.to_csv("datasetwinw/winequality_red_preparado.csv", index=False)

# Vinho Branco
X_white = vinhowhite_limpo.drop(columns=['quality'])
y_white = vinhowhite_limpo['quality']
scaler_white = StandardScaler()
X_white_scaled = scaler_white.fit_transform(X_white)

vinhowhite_preparado = pd.DataFrame(X_white_scaled, columns=X_white.columns)
vinhowhite_preparado['quality'] = y_white.values
vinhowhite_preparado.to_csv("datasetwinw/winequality_white_preparado.csv", index=False)
