# ==========================================
# QUESTÃO 7 - Definição do Problema de Regressão
# ==========================================
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from _dados_ import *

variavel_alvo = 'Y1'

variaveis_preditoras = ['X1', 'X2', 'X3', 'X4', 'X5', 'X7']

print("--- QUESTÃO 7 ---")
print("Variável Alvo escolhida:", variavel_alvo)
print("Variáveis preditoras:", variaveis_preditoras)
print("Aviso: A variável Y2 (Resfriamento) foi excluída para evitar vazamento de dados.")

correlacoes = energy[variaveis_preditoras + [variavel_alvo]].corr()[variavel_alvo].sort_values(ascending=False)
print("\n--- Correlação das preditoras com o Alvo (Y1) ---")
print(correlacoes)

plt.figure(figsize=(8, 6))
sns.heatmap(energy[variaveis_preditoras].corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Matriz de Correlação entre as Variáveis Preditoras (Multicolinearidade)')
plt.show()

import matplotlib.pyplot as plt
from _dados_ import *

variavel_alvo = 'Y1'
variaveis_preditoras = ['X1', 'X2', 'X3', 'X4', 'X5', 'X7'] # Sem X6 e X8

correlacoes = energy[variaveis_preditoras + [variavel_alvo]].corr()[variavel_alvo].drop(variavel_alvo).sort_values()

plt.figure(figsize=(10, 5))
correlacoes.plot(kind='barh', color='skyblue')
plt.title('Correlação das Variáveis com o Alvo (Y1)')
plt.xlabel('Índice de Correlação')
plt.axvline(x=0, color='black', linestyle='--')
plt.show()