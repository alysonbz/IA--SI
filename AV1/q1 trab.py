import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.base import clone
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score

# ==============================================================================
# QUESTÃO 1: PREPARAÇÃO DE DADOS, ESTATÍSTICAS E EXPORTAÇÃO
# ==============================================================================

vinhored = pd.read_csv("datasetwinw/winequality-red.csv", sep=";")
vinhowhite = pd.read_csv("datasetwinw/winequality-white.csv", sep=";")

# 1. DIMENSÕES, TIPOS DE DADOS E VARIÁVEL CLASSE
print("--- 1. DIMENSÕES E ESTRUTURA ---")
print("Dimensões Tinto (linhas, colunas):", vinhored.shape)
print("Dimensões Branco (linhas, colunas):", vinhowhite.shape)

print("\nTipos de dados (Tinto):\n", vinhored.dtypes)
print("\nTipos de dados (Branco):\n", vinhowhite.dtypes)

print("\nClasses Únicas ('quality') - Tinto:", sorted(vinhored['quality'].unique()))
print("Classes Únicas ('quality') - Branco:", sorted(vinhowhite['quality'].unique()))

# 2. VERIFICAÇÃO E TRATAMENTO DE INCONSISTÊNCIAS
print("\n--- 2. VERIFICAÇÃO DE INCONSISTÊNCIAS ---")
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
print("\n--- 3. IMPACTO DA REMOÇÃO DE DUPLICADAS ---")
print(f"Tinto  -> Antes: {vinhored.shape} | Depois: {vinhored_limpo.shape} | Removidos: {vinhored.shape[0] - vinhored_limpo.shape[0]}")
print(f"Branco -> Antes: {vinhowhite.shape} | Depois: {vinhowhite_limpo.shape} | Removidos: {vinhowhite.shape[0] - vinhowhite_limpo.shape[0]}")

# 4. ANÁLISE ESTATÍSTICA E EVIDÊNCIA DE ESCALAS
print("\n--- 4. AMPLITUDE DE ESCALAS ---")
atributos_red = vinhored_limpo.drop(columns=['quality'])
evidencia_escala_red = pd.DataFrame({
    'Minimo': atributos_red.min(),
    'Maximo': atributos_red.max(),
    'Amplitude (Max - Min)': atributos_red.max() - atributos_red.min()
}).round(3)
print("EVIDÊNCIA DE AMPLITUDES (TINTO):\n", evidencia_escala_red)

atributos_white = vinhowhite_limpo.drop(columns=['quality'])
evidencia_escala_white = pd.DataFrame({
    'Minimo': atributos_white.min(),
    'Maximo': atributos_white.max(),
    'Amplitude (Max - Min)': atributos_white.max() - atributos_white.min()
}).round(3)
print("\nEVIDÊNCIA DE AMPLITUDES (BRANCO):\n", evidencia_escala_white)

# 5. ANÁLISE DA DISTRIBUIÇÃO DAS CLASSES (COM %)
print("\n--- 5. DISTRIBUIÇÃO DAS CLASSES ---")
dist_red = pd.DataFrame({
    'Contagem': vinhored_limpo['quality'].value_counts().sort_index(),
    'Percentual (%)': (vinhored_limpo['quality'].value_counts(normalize=True).sort_index() * 100).round(2)
})
print("DISTRIBUIÇÃO DAS CLASSES (TINTO):\n", dist_red)

# 6. EXPORTAÇÃO DOS DATASETS E SEPARAÇÃO DE TREINO/TESTE
vinhored_limpo.to_csv("datasetwinw/winequality_red_limpo.csv", index=False)
vinhowhite_limpo.to_csv("datasetwinw/winequality_white_limpo.csv", index=False)

# Separando X e Y globais
X_red = vinhored_limpo.drop(columns=['quality'])
y_red = vinhored_limpo['quality']

X_white = vinhowhite_limpo.drop(columns=['quality'])
y_white = vinhowhite_limpo['quality']

# Divisão Correta com Estratificação (Para Q2 e Q3)
X_train_red, X_test_red, y_train_red, y_test_red = train_test_split(
    X_red, y_red, test_size=0.2, random_state=42, stratify=y_red
)
X_train_white, X_test_white, y_train_white, y_test_white = train_test_split(
    X_white, y_white, test_size=0.2, random_state=42, stratify=y_white
)

# Exportação do dataset padronizado global (Apenas para fins da questão 1)
scaler_export_red = StandardScaler()
vinhored_preparado = pd.DataFrame(scaler_export_red.fit_transform(X_red), columns=X_red.columns)
vinhored_preparado['quality'] = y_red.values
vinhored_preparado.to_csv("datasetwinw/winequality_red_preparado.csv", index=False)

scaler_export_white = StandardScaler()
vinhowhite_preparado = pd.DataFrame(scaler_export_white.fit_transform(X_white), columns=X_white.columns)
vinhowhite_preparado['quality'] = y_white.values
vinhowhite_preparado.to_csv("datasetwinw/winequality_white_preparado.csv", index=False)


# ==============================================================================
# QUESTÃO 2: COMPARAÇÃO DO KNN COM E SEM NORMALIZAÇÃO
# ==============================================================================
print("\n\n==================================================")
print("QUESTÃO 2: AVALIAÇÃO DE DESEMPENHO (ESCALAS)")
print("==================================================")

resultados = {}

# --- AVALIAÇÃO: SEM ESCALA ---
knn_r = KNeighborsClassifier()
knn_r.fit(X_train_red, y_train_red)
acc_red = knn_r.score(X_test_red, y_test_red)
acc_red_no_white = knn_r.score(X_test_white, y_test_white)

knn_w = KNeighborsClassifier()
knn_w.fit(X_train_white, y_train_white)
acc_white = knn_w.score(X_test_white, y_test_white)

resultados['Sem escala'] = (acc_red, acc_red_no_white, acc_white)
print("\n[Sem escala] Treino tinto -> teste tinto:", round(acc_red, 4))
print("[Sem escala] Treino tinto -> teste branco:", round(acc_red_no_white, 4))
print("[Sem escala] Treino branco -> teste branco:", round(acc_white, 4))

# Função auxiliar para evitar vazamento de dados
def avaliar_com_escala(nome, scaler):
    # --- Modelo do TINTO ---
    scaler_r = clone(scaler)
    X_train_red_s = scaler_r.fit_transform(X_train_red)   # fit + transform apenas no treino
    X_test_red_s = scaler_r.transform(X_test_red)         # apenas transform no teste
    X_test_white_s_red = scaler_r.transform(X_test_white) # teste branco no scaler tinto

    modelo_r = KNeighborsClassifier()
    modelo_r.fit(X_train_red_s, y_train_red)
    acc_r = modelo_r.score(X_test_red_s, y_test_red)
    acc_r_w = modelo_r.score(X_test_white_s_red, y_test_white)

    # --- Modelo do BRANCO ---
    scaler_w = clone(scaler)
    X_train_white_s = scaler_w.fit_transform(X_train_white)
    X_test_white_s = scaler_w.transform(X_test_white)

    modelo_w = KNeighborsClassifier()
    modelo_w.fit(X_train_white_s, y_train_white)
    acc_w = modelo_w.score(X_test_white_s, y_test_white)

    resultados[nome] = (acc_r, acc_r_w, acc_w)

    print(f"\n[{nome}] Treino tinto -> teste tinto:", round(acc_r, 4))
    print(f"[{nome}] Treino tinto -> teste branco:", round(acc_r_w, 4))
    print(f"[{nome}] Treino branco -> teste branco:", round(acc_w, 4))

avaliar_com_escala('StandardScaler', StandardScaler())
avaliar_com_escala('MinMaxScaler', MinMaxScaler())
avaliar_com_escala('RobustScaler', RobustScaler())

# --- COMPARAÇÃO FINAL ---
comparacao = pd.DataFrame(
    resultados,
    index=['Tinto -> Tinto', 'Tinto -> Branco', 'Branco -> Branco']
).T.round(4)

print("\n--- COMPARAÇÃO FINAL (KNN, k=5) ---")
print(comparacao)


# ==============================================================================
# QUESTÃO 3: BUSCA DO MELHOR K (HIPERPARÂMETRO)
# ==============================================================================
print("\n\n==================================================")
print("QUESTÃO 3: BUSCA PELO MELHOR VALOR DE K")
print("==================================================")

scalers_disponiveis = {
    'StandardScaler': StandardScaler,
    'MinMaxScaler': MinMaxScaler,
    'RobustScaler': RobustScaler,
}

# Escolhe a técnica com maior acurácia (Tinto -> Tinto) ignorando "Sem escala"
melhor_nome = comparacao.drop(index='Sem escala')['Tinto -> Tinto'].idxmax()
print(f"Técnica de normalização vencedora adotada na Questão 3: {melhor_nome}")

VALORES_K = range(1, 16)

def investigar_k(X_tr, y_tr, X_te, y_te, nome_scaler, titulo_vinho):
    scaler = scalers_disponiveis[nome_scaler]()
    X_tr_s = scaler.fit_transform(X_tr)
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
    print(f"Melhor K: {int(melhor['Valor de K'])} (validação = {melhor['Acurácia de validação']:.4f})")

    # Gráfico de linhas
    plt.figure(figsize=(9, 5))
    plt.plot(tabela['Valor de K'], tabela['Acurácia de treinamento'], marker='o', label='Acurácia de treinamento')
    plt.plot(tabela['Valor de K'], tabela['Acurácia de validação'], marker='s', label='Acurácia de validação')
    plt.axvline(melhor['Valor de K'], color='gray', linestyle='--', alpha=0.6, label=f"Melhor K = {int(melhor['Valor de K'])}")
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

# Roda a investigação para Tinto e Branco
tabela_k_red = investigar_k(X_train_red, y_train_red, X_test_red, y_test_red, melhor_nome, "tinto")
tabela_k_white = investigar_k(X_train_white, y_train_white, X_test_white, y_test_white, melhor_nome, "branco")

# ==============================================================================
# QUESTÃO 3 (COMPLEMENTO): K ESCOLHIDO SEM USAR O TESTE + MODELO FINAL
# ==============================================================================

print("\n\n==================================================")
print("QUESTÃO 3 (COMPLEMENTO): K ESCOLHIDO SEM USAR O TESTE + MODELO FINAL")
print("==================================================")

# Separa uma validação de verdade DENTRO do treino (o teste fica intocado)
X_tr3, X_val3, y_tr3, y_val3 = train_test_split(
    X_train_red, y_train_red, test_size=0.25, random_state=42, stratify=y_train_red
)
cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

linhas = []
for k in range(1, 31):
    pipe = make_pipeline(scalers_disponiveis[melhor_nome](),
                         KNeighborsClassifier(n_neighbors=k, metric='euclidean'))
    pipe.fit(X_tr3, y_tr3)
    linhas.append({
        'Valor de K': k,
        'Acurácia de treinamento': pipe.score(X_tr3, y_tr3),
        'Acurácia de validação': pipe.score(X_val3, y_val3),
        'Acurácia CV-5': cross_val_score(pipe, X_train_red, y_train_red, cv=cv5).mean(),
    })
tabela_q3 = pd.DataFrame(linhas).round(4)
print(tabela_q3.to_string(index=False))

melhor_val = tabela_q3.loc[tabela_q3['Acurácia de validação'].idxmax()]
melhor_cv = tabela_q3.loc[tabela_q3['Acurácia CV-5'].idxmax()]
k_final = int(melhor_cv['Valor de K'])
print(f"Maior acurácia de validação: K={int(melhor_val['Valor de K'])} ({melhor_val['Acurácia de validação']:.4f})")
print(f"Maior acurácia CV-5 (critério adotado): K={k_final} ({melhor_cv['Acurácia CV-5']:.4f})")
print("K a até 1 p.p. do melhor na CV-5:",
      tabela_q3[tabela_q3['Acurácia CV-5'] >= melhor_cv['Acurácia CV-5'] - 0.01]['Valor de K'].tolist())

plt.figure(figsize=(10, 5.5))
plt.plot(tabela_q3['Valor de K'], tabela_q3['Acurácia de treinamento'], marker='o', label='Acurácia de treinamento')
plt.plot(tabela_q3['Valor de K'], tabela_q3['Acurácia de validação'], marker='s', label='Acurácia de validação')
plt.axvline(k_final, color='gray', linestyle='--', alpha=0.6, label=f'K escolhido = {k_final}')
plt.title(f"KNN - acurácia por K (vinho tinto, {melhor_nome}) - validação sem usar o teste")
plt.xlabel("Valor de K"); plt.ylabel("Acurácia")
plt.grid(alpha=0.3); plt.legend(); plt.tight_layout()
plt.savefig("datasetwinw/knn_k_tinto_validacao.png", dpi=150)
plt.show()

# MODELO FINAL: treina com todo o treino e usa o TESTE uma única vez
final = make_pipeline(scalers_disponiveis[melhor_nome](),
                      KNeighborsClassifier(n_neighbors=k_final, metric='euclidean'))
final.fit(X_train_red, y_train_red)
print(f"\nMODELO FINAL: {melhor_nome} + K={k_final}")
print(f"Acurácia no conjunto de TESTE: {final.score(X_test_red, y_test_red):.4f}")