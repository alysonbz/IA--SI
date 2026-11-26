# ==========================================
# QUESTÃO 8 - Preparação dos dados para regressão
# ==========================================
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from _dados_ import * # Traz o dataset 'energy'

variavel_alvo = 'Y1'
variaveis_preditoras = ['X1', 'X2', 'X3', 'X4', 'X5', 'X7']

X_dados = energy[variaveis_preditoras]
y_dados = energy[variavel_alvo]

X_train, X_test, y_train, y_test = train_test_split(
    X_dados, y_dados, test_size=0.20, random_state=42
)
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)

X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=variaveis_preditoras, index=X_train.index)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=variaveis_preditoras, index=X_test.index)

# --- Prints de verificação ---
print("--- QUESTÃO 8 ---")
print(f"Total original: {len(energy)} amostras")
print(f"Treino (80%): {len(X_train)} amostras")
print(f"Teste (20%): {len(X_test)} amostras")
print("\nMédia das preditoras no Treino (deve ser ~0):")
print(X_train_scaled_df.mean().round(2))
print("\nDesvio padrão no Treino (deve ser ~1):")
print(X_train_scaled_df.std().round(2))