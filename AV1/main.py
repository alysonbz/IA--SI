import numpy as np
import pandas as pd  # tabelas (DataFrame)
import matplotlib.pyplot as plt  # gráficos
from pathlib import Path  # para checar se um arquivo existe
import os  # para fixar a pasta de trabalho

from sklearn.model_selection import train_test_split
# train_test_split: divide os dados em treino/validação/teste
from sklearn.preprocessing import MinMaxScaler, StandardScaler, FunctionTransformer
# as 3 técnicas que vamos comparar: MinMax, Standard e a transformação logarítmica
# (FunctionTransformer permite usar uma função nossa, no caso o log, dentro do Pipeline)
from sklearn.compose import ColumnTransformer  # aplica o scaler só em certas colunas
from sklearn.pipeline import Pipeline  # encadeia "scaler -> KNN" num objeto só
from sklearn.neighbors import KNeighborsClassifier  # o KNN
from sklearn.metrics import accuracy_score, f1_score, recall_score
# ---------------------------------------------------------------------
# 1.5 Escalas das variáveis (para decidir sobre normalização)
# ---------------------------------------------------------------------
titulo("Q1.5 - Escalas das variáveis")
amplitude = (df[entradas].max() - df[entradas].min()).sort_values()  # max - min de cada coluna
print(amplitude)
print(f"\nMaior amplitude / menor amplitude = {amplitude.max() / amplitude.min():,.0f}x")

print("""
DECISÃO: o KNN mede distância euclidiana = raiz(soma das diferenças ao quadrado).
Variáveis com valores enormes (LIMIT_BAL, BILL_AMT, PAY_AMT) dominam essa soma, e as
variáveis de atraso (PAY_1..PAY_6, de -2 a 8), que são as mais informativas, quase não
pesam. O KNN não corrige isso sozinho -> normalizar/padronizar é NECESSÁRIO.
""")

# ---------------------------------------------------------------------
# 1.6 Tratamento (gera o dataset da Questão 2)
# ---------------------------------------------------------------------
titulo("Q1.6 - Tratamento aplicado")
dim_antes = df.shape
dfc = df.copy()  # trabalhamos numa cópia, o original fica intacto

# (a) EDUCATION: 0, 5, 6 não existem na documentação -> agrupados em 4 ("outros").
#     Agrupar em vez de apagar mantém as linhas (menos perda de dados).
dfc["EDUCATION"] = dfc["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
# (b) MARRIAGE: 0 não documentado -> 3 ("outros").
dfc["MARRIAGE"] = dfc["MARRIAGE"].replace({0: 3})
# (c) SEX vira 0/1 (1=masculino, 2=feminino -> 0/1). Uma binária pode ficar numa coluna só.
dfc["SEX"] = (dfc["SEX"] == 2).astype(int)
# (d) Duplicatas removidas: repetem uma amostra e inflam a acurácia (o vizinho idêntico "entrega" a resposta).
n_dup = int(dfc.duplicated().sum())
dfc = dfc.drop_duplicates().reset_index(drop=True)
# (e) One-hot em EDUCATION e MARRIAGE: são categorias SEM ordem. Deixar 1,2,3,4 faria o KNN achar
#     que "1" está mais perto de "2" do que de "4". Com colunas 0/1 todas ficam equidistantes.
dfc = pd.get_dummies(dfc, columns=["EDUCATION", "MARRIAGE"], dtype=int)
# Mantidos de propósito: fatura negativa (valor real), PAY_x = -2/0 (ordinais plausíveis) e
# outliers (são clientes reais; quem lida com eles é o scaler na Q2).

print("Duplicatas removidas:", n_dup)
print("Dimensões antes :", dim_antes)
print("Dimensões depois:", dfc.shape)
dfc.to_csv("dataset_preparado_questao2.csv", index=False)  # versão final pedida no enunciado
print("Salvo em dataset_preparado_questao2.csv")

# =====================================================================
# QUESTÃO 2 — KNN COM E SEM NORMALIZAÇÃO
# =====================================================================
titulo("Q2 - Comparação das normalizações")

X = dfc.drop(columns="DEFAULT")  # entradas
y = dfc["DEFAULT"]  # classe

# Colunas binárias (0/1) não precisam de scaler; só escalamos as demais.
colunas_bin = [c for c in X if set(X[c].unique()) <= {0, 1}]
colunas_num = [c for c in X if c not in colunas_bin]
print("Colunas escaladas:", len(colunas_num), "| binárias mantidas:", len(colunas_bin))

# Divisão 60% treino / 20% validação / 20% teste.
# stratify=y mantém a mesma proporção de classes em cada parte.
# O MESMO split vale para todas as configurações: assim a comparação é justa.
X_dev, X_test, y_dev, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE)  # separa 20% para teste final
X_train, X_val, y_train, y_val = train_test_split(
    X_dev, y_dev, test_size=0.25, stratify=y_dev, random_state=RANDOM_STATE)  # 25% de 80% = 20% total
print("Tamanhos treino / validação / teste:", len(X_train), len(X_val), len(X_test))


# O teste fica GUARDADO e só será usado no final da Questão 3.


def log_assinado(x):
    """Transformação logarítmica que aceita valores negativos e zero.
    O log comum não existe para x <= 0, e BILL_AMT tem faturas negativas (crédito a favor do cliente).
    Por isso uso: sinal(x) * log(1 + |x|). Valores grandes são comprimidos (ex.: 800000 vira ~13,6),
    o que reduz a assimetria e o peso dos outliers; o sinal é preservado."""
    return np.sign(x) * np.log1p(np.abs(x))


def monta_modelo(scaler=None, k=5):
    """Cria um Pipeline: (scaler opcional) -> KNN.
    O Pipeline ajusta o scaler SÓ com os dados de treino e depois aplica o mesmo
    ajuste na validação. Isso evita 'vazamento' (o modelo espiar dados que não deveria)."""
    etapas = []
    if scaler is not None:
        etapas.append(("scaler", ColumnTransformer(
            [("esc", scaler, colunas_num)],  # aplica o scaler só nas colunas numéricas
            remainder="passthrough")))  # as binárias passam sem alteração
    etapas.append(("knn", KNeighborsClassifier(
        n_neighbors=k, metric="euclidean", n_jobs=-1)))  # n_jobs=-1: usa todos os núcleos
    return Pipeline(etapas)


# As 4 configurações. Cada valor é uma "fábrica" que cria um scaler novo a cada uso.
configs = {
    "Sem normalização": lambda: None,
    "MinMaxScaler (0-1)": MinMaxScaler,  # (x - min) / (max - min)
    "StandardScaler (z-score)": StandardScaler,  # (x - média) / desvio
    "Logarítmica (log1p com sinal)": lambda: FunctionTransformer(log_assinado),  # sinal(x)*log(1+|x|)
}

linhas = []
for i, (nome, fabrica) in enumerate(configs.items(), start=1):
    modelo = monta_modelo(fabrica(), k=5).fit(X_train, y_train)  # treina (k=5 fixo para todas)
    pred = modelo.predict(X_val)  # prevê a validação
    linhas.append({
        "Configuração": i,
        "Técnica utilizada": nome,
        "Acurácia (validação)": accuracy_score(y_val, pred),  # % de acertos
        "Recall classe 1": recall_score(y_val, pred),  # dos que NÃO pagaram, quantos achei
        "F1 classe 1": f1_score(y_val, pred),  # equilíbrio entre precisão e recall
    })
tab2 = pd.DataFrame(linhas)
print(tab2.round(4).to_string(index=False))

p = tab2["Acurácia (validação)"].max()
margem = 1.96 * np.sqrt(p * (1 - p) / len(y_val))  # margem de erro de ~95% para uma acurácia
print(f"\nBaseline da classe majoritária: {baseline:.4f}")
print(f"Margem de erro (95%) da acurácia na validação: +-{margem:.4f}")
print("-> Diferenças menores que essa margem NÃO provam superioridade de uma técnica.")

melhor = tab2.sort_values("Acurácia (validação)", ascending=False).iloc[0]
print("\nConfiguração com maior acurácia na validação:", melhor["Técnica utilizada"])
ganho = (tab2["Acurácia (validação)"] - tab2.loc[0, "Acurácia (validação)"]) * 100
print("Ganho em pontos percentuais sobre 'Sem normalização':")
print(pd.Series(ganho.round(2).values, index=tab2["Técnica utilizada"]).to_string())

# =====================================================================
# QUESTÃO 3 — ESCOLHA DO VALOR DE K
# =====================================================================
titulo("Q3 - Escolha de K")

ESCOLHIDA = melhor["Técnica utilizada"]  # normalização vencedora da Q2
# ESCOLHIDA = "StandardScaler (z-score)"    # descomente para forçar outra
fabrica = configs[ESCOLHIDA]
print("Técnica usada:", ESCOLHIDA)

ks = list(range(1, 42, 2))  # 21 valores ímpares (1,3,...,41). Ímpar evita empate na votação.
resultados = []
for k in ks:
    modelo = monta_modelo(fabrica(), k=k).fit(X_train, y_train)
    resultados.append({
        "K": k,
        "Acurácia de treinamento": accuracy_score(y_train, modelo.predict(X_train)),  # nos dados que ele viu
        "Acurácia de validação": accuracy_score(y_val, modelo.predict(X_val)),  # em dados novos
    })
tab3 = pd.DataFrame(resultados)
print(tab3.round(4).to_string(index=False))

# Gráfico: treino x validação. A DISTÂNCIA entre as linhas mostra o sobreajuste.
plt.figure(figsize=(9, 5))
plt.plot(tab3["K"], tab3["Acurácia de treinamento"], "o-", label="Treinamento")
plt.plot(tab3["K"], tab3["Acurácia de validação"], "s-", label="Validação")
plt.xlabel("Valor de K")
plt.ylabel("Acurácia")
plt.title(f"KNN: acurácia em função de K ({ESCOLHIDA})")
plt.xticks(ks)
plt.grid(alpha=0.3)
plt.legend()
plt.savefig("q3_k_vs_acuracia.png", dpi=120, bbox_inches="tight")
plt.show()

# Escolha do K SEM olhar o teste.
melhor_k = tab3.loc[tab3["Acurácia de validação"].idxmax()]
acc = melhor_k["Acurácia de validação"]
erro_padrao = np.sqrt(acc * (1 - acc) / len(y_val))  # incerteza típica dessa acurácia
proximos = tab3[tab3["Acurácia de validação"] >= acc - erro_padrao]
print(f"\nMaior acurácia de validação: K={int(melhor_k['K'])} ({acc:.4f}); erro-padrão ~ {erro_padrao:.4f}")
print("Ks estatisticamente equivalentes (dentro de 1 erro-padrão):", proximos["K"].tolist())
# Regra do 1 erro-padrão: entre os equivalentes, escolho o MAIOR K (modelo mais estável/suave).
K_FINAL = int(proximos["K"].max())
print("K escolhido:", K_FINAL)

# ---------------------------------------------------------------------
# Modelo final: única vez que o conjunto de TESTE é usado
# ---------------------------------------------------------------------
final = monta_modelo(fabrica(), k=K_FINAL).fit(X_dev, y_dev)  # treina com treino+validação
pred_teste = final.predict(X_test)
print(f"\nK={K_FINAL} | {ESCOLHIDA}")
print(f"Acurácia no TESTE: {accuracy_score(y_test, pred_teste):.4f}  (baseline: {baseline:.4f})")
print(f"Recall classe 1: {recall_score(y_test, pred_teste):.4f} | F1 classe 1: {f1_score(y_test, pred_teste):.4f}")
