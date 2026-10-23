import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# 1. CARREGANDO O DATASET
df = pd.read_csv("risk_factors_cervical_cancer.csv")
df.columns = df.columns.str.strip()

print("\nPRIMEIRAS LINHAS ")
print(df.head())

print("\nDIMENSÕES")
print(df.shape)

print("\nCOLUNAS ")
print(df.columns.tolist())

print("\nTIPOS ")
print(df.dtypes)

print("\nINFORMAÇÕES ")
df.info()

# 2. VERIFICANDO INCONSISTÊNCIAS E TRATAMENTO INICIAL
# No dataset, "?" representa valor ausente
df = df.replace("?", pd.NA)

print("\nVALORES AUSENTES ")
print(df.isna().sum())

print("\nDUPLICATAS ")
print(df.duplicated().sum())

# Transformar dados em números
for coluna in df.columns:
    df[coluna] = pd.to_numeric(df[coluna], errors="coerce")


# 3. ESTATÍSTICAS DESCRITIVAS
print("\nESTATÍSTICAS DESCRITIVAS ")
print(df.describe())

# 4. DISTRIBUIÇÃO DAS CLASSES
print("\nDISTRIBUIÇÃO DE Dx:Cancer")
print(df["Dx:Cancer"].value_counts())

print("\nPORCENTAGEM DAS CLASSES")
print(df["Dx:Cancer"].value_counts(normalize=True) * 100)

# Gráfico da distribuição das classes
df["Dx:Cancer"].value_counts().plot(kind="bar", color=["skyblue", "salmon"])
plt.title("Distribuição da Classe Target (Dx:Cancer)")
plt.xlabel("Classe (0 = Não, 1 = Sim)")
plt.ylabel("Quantidade de Amostras")
plt.xticks(rotation=0)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()

# 5. DEFININDO FEATURES (X) E TARGET (y)
y = df["Dx:Cancer"]

# Ajuste automático para aceitar com ou sem espaço após os dois pontos
colunas_disponiveis = df.columns.tolist()
std_hiv_col = "STDs:HIV" if "STDs:HIV" in colunas_disponiveis else "STDs: HIV"
std_hpv_col = "STDs:HPV" if "STDs:HPV" in colunas_disponiveis else "STDs: HPV"

features = [
    "Age",
    "Number of sexual partners",
    "First sexual intercourse",
    "Num of pregnancies",
    "Smokes",
    "Smokes (years)",
    "Smokes (packs/year)",
    "Hormonal Contraceptives",
    "Hormonal Contraceptives (years)",
    "STDs",
    std_hiv_col,
    std_hpv_col,
]

X = df[features]

print("\nFEATURES SELECIONADAS ")
print("Quantidade de atributos:", X.shape[1])

# 6. REMOVENDO TARGET AUSENTE
dados_validos = y.notna()
X = X[dados_validos]
y = y[dados_validos]


# 7. DIVISÃO EM TREINO E TESTE (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print("\nPARTIÇÃO TREINO E TESTE ")
print("Treino Completo:", X_train.shape)
print("Teste Reservado:", X_test.shape)


# 8. DIVISÃO EM TREINO E VALIDAÇÃO (80/20 do Treino)
X_treino, X_validacao, y_treino, y_validacao = train_test_split(
    X_train, y_train, test_size=0.20, random_state=42, stratify=y_train
)

print("\nPARTIÇÃO TREINO E VALIDAÇÃO")
print("Treino (Sub-partição):", X_treino.shape)
print("Validação:", X_validacao.shape)

# 9. IMPUTAÇÃO DOS VALORES AUSENTES (MEDIANA)
imputer = SimpleImputer(strategy="median")
X_treino_imp = imputer.fit_transform(X_treino)
X_validacao_imp = imputer.transform(X_validacao)


# QUESTÃO 2: COMPARAÇÃO DAS TÉCNICAS DE NORMALIZAÇÃO (SLIDES 8 E 9)

print("QUESTÃO 2: COMPARAÇÃO DE NORMALIZAÇÃO (K = 5)")

configuracoes = {
    "1. Sem normalização": None,
    "2. StandardScaler (Z-score)": StandardScaler(),
    "3. MinMaxScaler (0 a 1)": MinMaxScaler(),
    "4. RobustScaler (IQR)": RobustScaler(),
}

print(f"{'Configuração':<30} | {'Técnica Utilizada':<30} | {'Acurácia Val.':<15}")
print("-" * 80)

for idx, (nome_config, scaler_obj) in enumerate(configuracoes.items(), 1):
    if scaler_obj is not None:
        X_tr_sc = scaler_obj.fit_transform(X_treino_imp)
        X_val_sc = scaler_obj.transform(X_validacao_imp)
    else:
        X_tr_sc = X_treino_imp
        X_val_sc = X_validacao_imp

    knn_q2 = KNeighborsClassifier(n_neighbors=5)
    knn_q2.fit(X_tr_sc, y_treino)
    acc_q2 = accuracy_score(y_validacao, knn_q2.predict(X_val_sc)) * 100
    print(f"{idx:<30} | {nome_config:<30} | {acc_q2:6.2f}%")

# Padronização padrão para a Questão 3 (StandardScaler)
scaler = StandardScaler()
X_treino_sc = scaler.fit_transform(X_treino_imp)
X_validacao_sc = scaler.transform(X_validacao_imp)


# QUESTÃO 3: INVESTIGAÇÃO DO HIPERPARÂMETRO K (SLIDES 14, 15 E 16)

valores_k = list(range(1, 20, 2))

acc_treino_list = []
acc_valid_list = []

print("QUESTÃO 3: TABELA DE K x ACURÁCIA")
print("| Valor K  | Acurácia de Treino    | Acurácia de Validação    |")

for k in valores_k:
    modelo = KNeighborsClassifier(n_neighbors=k)
    modelo.fit(X_treino_sc, y_treino)

    # Acurácia de Treino
    pred_tr = modelo.predict(X_treino_sc)
    acc_tr = accuracy_score(y_treino, pred_tr) * 100
    acc_treino_list.append(acc_tr)

    # Acurácia de Validação
    pred_val = modelo.predict(X_validacao_sc)
    acc_val = accuracy_score(y_validacao, pred_val) * 100
    acc_valid_list.append(acc_val)

    print(f"| K = {k:<4} | {acc_tr:6.2f}%               | {acc_val:6.2f}%                   |")

# Escolha do melhor K baseado na acurácia de validação
melhor_posicao = acc_valid_list.index(max(acc_valid_list))
melhor_k = valores_k[melhor_posicao]

print("\nMELHOR K SELECIONADO")
print("K escolhido:", melhor_k)
print("Maior Acurácia na validação:", round(max(acc_valid_list), 2), "%")


# GRÁFICO DA CURVA DE APRENDIZADO: K x ACURÁCIA (SLIDE 16)
plt.figure(figsize=(8, 5))
plt.plot(valores_k, acc_treino_list, marker="o", linestyle="-", color="tab:blue", label="Acurácia de Treino")
plt.plot(valores_k, acc_valid_list, marker="s", linestyle="--", color="tab:orange", label="Acurácia de Validação")

plt.xlabel("Valor de K")
plt.ylabel("Acurácia (%)")
plt.title("Curva de Aprendizado: Valor de K vs. Acurácia")
plt.xticks(valores_k)
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()


# PREPARANDO O TREINO COMPLETO E AVALIAÇÃO NO TESTE RESERVADO (SLIDES 19 E 20)
imputer_final = SimpleImputer(strategy="median")
X_train_imp = imputer_final.fit_transform(X_train)
X_test_imp = imputer_final.transform(X_test)

scaler_final = StandardScaler()
X_train_sc = scaler_final.fit_transform(X_train_imp)
X_test_sc = scaler_final.transform(X_test_imp)

# Treinamento do modelo final com K escolhido
knn_final = KNeighborsClassifier(n_neighbors=melhor_k)
knn_final.fit(X_train_sc, y_train)

# Previsões no conjunto de teste
y_pred = knn_final.predict(X_test_sc)


# AVALIAÇÃO FINAL NO CONJUNTO DE TESTE
acuracia_final = accuracy_score(y_test, y_pred)
f1_final = f1_score(y_test, y_pred, zero_division=0)

print("RESULTADO FINAL NO CONJUNTO DE TESTE")
print("K Escolhido:", melhor_k)
print("Acurácia Final:", round(acuracia_final, 4))
print("Acurácia Final (%):", round(acuracia_final * 100, 2), "%")
print("F1-Score Final:", round(f1_final, 4))

print("\nMATRIZ DE CONFUSÃO ")
matriz = confusion_matrix(y_test, y_pred)
print(matriz)

print("\nRELATÓRIO DE CLASSIFICAÇÃO")
print(classification_report(y_test, y_pred, zero_division=0))