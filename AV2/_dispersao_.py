import matplotlib.pyplot as plt
import seaborn as sns
from _dados_ import *

# Gráfico de dispersão
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Gráficos de Dispersão: Relações entre Características', fontsize=16, fontweight='bold')

# Par 1: Área de Superfície vs Altura (Geralmente mostram relação inversa)
sns.scatterplot(data=energy, x='X2', y='X5', ax=axes[0], alpha=0.6, color='coral')
axes[0].set_title('Área de Superfície (X2) vs Altura (X5)')
axes[0].set_xlabel('Área de Superfície (X2)')
axes[0].set_ylabel('Altura (X5)')

# Par 2: Compacidade vs Área de Superfície (Altamente relacionadas)
sns.scatterplot(data=energy, x='X1', y='X2', ax=axes[1], alpha=0.6, color='teal')
axes[1].set_title('Compacidade (X1) vs Área de Superfície (X2)')
axes[1].set_xlabel('Compacidade Relativa (X1)')
axes[1].set_ylabel('Área de Superfície (X2)')

# Par 3: Área Envidraçada vs Área de Parede
sns.scatterplot(data=energy, x='X7', y='X3', ax=axes[2], alpha=0.6, color='purple')
axes[2].set_title('Área Envidraçada (X7) vs Área de Parede (X3)')
axes[2].set_xlabel('Área Envidraçada (X7)')
axes[2].set_ylabel('Área de Parede (X3)')

plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# 4. MATRIZ DE CORRELAÇÃO (Heatmap)
# ---------------------------------------------------------
# Selecionando todas as variáveis numéricas, incluindo os alvos (Y1 e Y2) para análise futura
variaveis_corr = ["X1", "X2", "X3", "X4", "X5", "X7", "Y1", "Y2"]

# Calculando a matriz de correlação
matriz_corr = energy[variaveis_corr].corr()

# Criando a figura
plt.figure(figsize=(10, 8))
sns.heatmap(matriz_corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, vmin=-1, vmax=1)
plt.title('Matriz de Correlação das Características', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()