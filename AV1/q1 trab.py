import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.base import clone
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler  # RobustScaler é novo
import matplotlib.pyplot as plt

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
print("\nEVIDÊNCIA DE AMPLITUDES (TINTO)")
print(evidencia_escala)  # corrigido: antes imprimia o texto "evidencia_escala" em vez da tabela

# 5. ANÁLISE DA DISTRIBUIÇÃO DAS CLASSES (COM %)
print("\nDISTRIBUIÇÃO DAS CLASSES (TINTO)")
dist_red = pd.DataFrame({
    'Contagem': vinhored_limpo['quality'].value_counts().sort_index(),
    'Percentual (%)': (vinhored_limpo['quality'].value_counts(normalize=True).sort_index() * 100).round(2)
})
print(dist_red)

# 6. EXPORTAÇÃO DOS DATASETS
# Versão 1: Escala original sem duplicados
vinhored_limpo.to_csv("datasetwinw/winequality_red_limpo.csv", index=False)
vinhowhite_limpo.to_csv("datasetwinw/winequality_white_limpo.csv", index=False)

# Vinho Tinto
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

# Questão 2:
# ==================================================================
# Dados sem normalização ou padronização;
# ==================================================================
X_train, X_test, y_train, y_test = train_test_split(X_red, y_red, test_size=0.2, random_state=42)
X_train_w, X_test_w, y_train_w, y_test_w = train_test_split(X_white, y_white, test_size=0.2, random_state=42)

print("\nDados do dataset red (treino):", X_train.shape, y_train.shape)
print("Dados do dataset white (teste):", X_test_w.shape, y_test_w.shape)

resultados = {}  # guarda as acurácias para a tabela final

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)

acc_red = knn.score(X_test, y_test)
acc_red_no_white = knn.score(X_test_w, y_test_w)

# Modelo treinado com o vinho BRANCO (usa X_train_w / y_train_w)
knn_w = KNeighborsClassifier()
knn_w.fit(X_train_w, y_train_w)
acc_white = knn_w.score(X_test_w, y_test_w)

resultados['Sem escala'] = (acc_red, acc_red_no_white, acc_white)
print("\n[Sem escala] Treino tinto -> teste tinto:", round(acc_red, 4))
print("[Sem escala] Treino tinto -> teste branco:", round(acc_red_no_white, 4))
print("[Sem escala] Treino branco -> teste branco:", round(acc_white, 4))


# Função auxiliar: cada scaler aprende SÓ com o treino do respectivo vinho
# (evita data leakage) e depois é aplicado aos conjuntos de teste.
def avaliar_com_escala(nome, scaler):
    # --- Modelo do TINTO ---
    X_train_s = scaler.fit_transform(X_train)   # fit + transform apenas no treino tinto
    X_test_s = scaler.transform(X_test)         # só transform no teste tinto
    X_test_w_s_red = scaler.transform(X_test_w) # teste branco, com o scaler do tinto

    modelo_r = KNeighborsClassifier()
    modelo_r.fit(X_train_s, y_train)
    acc_r = modelo_r.score(X_test_s, y_test)
    acc_r_w = modelo_r.score(X_test_w_s_red, y_test_w)

    # --- Modelo do BRANCO (scaler novo, ajustado só no treino branco) ---
    scaler_w = clone(scaler)
    X_train_w_s = scaler_w.fit_transform(X_train_w)
    X_test_w_s = scaler_w.transform(X_test_w)

    modelo_w = KNeighborsClassifier()
    modelo_w.fit(X_train_w_s, y_train_w)
    acc_w = modelo_w.score(X_test_w_s, y_test_w)

    resultados[nome] = (acc_r, acc_r_w, acc_w)

    print(f"[{nome}] Treino tinto -> teste tinto:", round(acc_r, 4))
    print(f"[{nome}] Treino tinto -> teste branco:", round(acc_r_w, 4))
    print(f"[{nome}] Treino branco -> teste branco:", round(acc_w, 4))


# ===========================================================================================================
# Dados transformados pela primeira técnica escolhida;
# =====================StandardScaler============================
# média 0 e desvio padrão 1 em cada atributo
print()
avaliar_com_escala('StandardScaler', StandardScaler())

# ===========================================================================================================
# Dados transformados pela segunda técnica escolhida;
# ======================Min-Max===================================
# leva cada atributo para o intervalo [0, 1]
print()
avaliar_com_escala('MinMaxScaler', MinMaxScaler())

# ===========================================================================================================
# Dados transformados pela terceira técnica escolhida.
# ======================RobustScaler==============================
# usa mediana e IQR (Q3 - Q1), então é pouco afetado pelos outliers
# (comuns em chlorides, sulphates, total sulfur dioxide etc.)
print()
avaliar_com_escala('RobustScaler', RobustScaler())

# ===========================================================================================================
# COMPARAÇÃO FINAL
# ===========================================================================================================
comparacao = pd.DataFrame(
    resultados,
    index=['Tinto -> Tinto', 'Tinto -> Branco', 'Branco -> Branco']
).T.round(4)
print("\nCOMPARAÇÃO FINAL (KNN, k=5)")
print(comparacao)


# ===========================================================================================================
# QUESTÃO 3: Melhor valor de K usando a melhor normalização da Questão 2
# ===========================================================================================================


scalers_disponiveis = {
    'StandardScaler': StandardScaler,
    'MinMaxScaler': MinMaxScaler,
    'RobustScaler': RobustScaler,
}

# Escolhe automaticamente a técnica com maior acurácia (Tinto -> Tinto) na Questão 2.
# Se quiser forçar uma, troque pela linha comentada abaixo.
melhor_nome = comparacao.drop(index='Sem escala')['Tinto -> Tinto'].idxmax()
# melhor_nome = 'StandardScaler'
print(f"\nTécnica de normalização adotada na Questão 3: {melhor_nome}")

VALORES_K = range(1, 16)


def investigar_k(X_tr, y_tr, X_te, y_te, nome_scaler, titulo_vinho):
    # Mesmo scaler, mesma métrica e mesma divisão treino/validação para todos os K
    scaler = scalers_disponiveis[nome_scaler]()
    X_tr_s = scaler.fit_transform(X_tr)   # fit só no treino
    X_te_s = scaler.transform(X_te)

    linhas = []
    for k in VALORES_K:
        modelo = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
        modelo.fit(X_tr_s, y_tr)
        linhas.append({
            'Valor de K': k,
            'Acurácia de treinamento': modelo.score(X_tr_s, y_tr),
            'Acurácia de validação': modelo.score(X_te_s, y_te),
        })

    tabela = pd.DataFrame(linhas).round(4)
    print(f"\nTABELA DE RESULTADOS - VINHO {titulo_vinho.upper()} ({nome_scaler})")
    print(tabela.to_string(index=False))

    melhor = tabela.loc[tabela['Acurácia de validação'].idxmax()]
    print(f"Melhor K: {int(melhor['Valor de K'])} "
          f"(validação = {melhor['Acurácia de validação']:.4f})")

    # Gráfico de linhas
    plt.figure(figsize=(9, 5))
    plt.plot(tabela['Valor de K'], tabela['Acurácia de treinamento'],
             marker='o', label='Acurácia de treinamento')
    plt.plot(tabela['Valor de K'], tabela['Acurácia de validação'],
             marker='s', label='Acurácia de validação')
    plt.axvline(melhor['Valor de K'], color='gray', linestyle='--', alpha=0.6,
                label=f"Melhor K = {int(melhor['Valor de K'])}")
    plt.title(f"KNN - Acurácia por valor de K (vinho {titulo_vinho}, {nome_scaler})")
    plt.xlabel("Valor de K")
    plt.ylabel("Acurácia")
    plt.xticks(list(VALORES_K))
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"datasetwinw/knn_k_{titulo_vinho}.png", dpi=150)
    plt.show()

    return tabela


tabela_k_red = investigar_k(X_train, y_train, X_test, y_test, melhor_nome, "tinto")
tabela_k_white = investigar_k(X_train_w, y_train_w, X_test_w, y_test_w, melhor_nome, "branco")