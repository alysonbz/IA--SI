# ==========================================
# QUESTÃO 7 - Definição do Problema de Regressão
# ==========================================
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from _dados_ import * # Traz o dataset 'energy'

# a) Selecionar a variável dependente (alvo) e justificar sua escolha
# Escolha: Y1 (Carga de Aquecimento)
# Justificativa: É o principal indicador de gasto energético em climas frios.
variavel_alvo = 'Y1'

# b) Identificar as variáveis independentes (apenas as usadas pela Laísa)
# Justificativa: X6 e X8 são variáveis categóricas (ex: orientação 1=Norte, 2=Sul).
# Se usadas como números, o modelo daria peso errado (ex: achar que 5 é maior que 1).
# X6 e X8 foram ignoradas para manter a consistência e evitar pesos irreais.
variaveis_preditoras = ['X1', 'X2', 'X3', 'X4', 'X5', 'X7']

# e) Identificar vazamento de dados (Data Leakage)
# A variável Y2 (Resfriamento) NÃO está na lista de preditoras.
# Motivo: Se usada, o modelo usaria a "resposta" para adivinhar o "alvo", o que é trapaça.
print("--- QUESTÃO 7 ---")
print("Variável Alvo escolhida:", variavel_alvo)
print("Variáveis preditoras:", variaveis_preditoras)
print("Aviso: A variável Y2 (Resfriamento) foi excluída para evitar vazamento de dados.")

# c) Analisar a relação entre a variável alvo e as características selecionadas
# Calculando a correlação de cada variável preditora com o alvo Y1.
correlacoes = energy[variaveis_preditoras + [variavel_alvo]].corr()[variavel_alvo].sort_values(ascending=False)
print("\n--- Correlação das preditoras com o Alvo (Y1) ---")
print(correlacoes)

# d) Verificar a presença de correlações elevadas entre as variáveis independentes
# Gerando o Heatmap para visualizar a multicolinearidade.
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