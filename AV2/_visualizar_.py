import matplotlib.pyplot as plt
import seaborn as sns
from _dados_ import *

sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)

variaveis_num = ["X1", "X2", "X3", "X4", "X5", "X7"]

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Distribuição das Principais Características (Histogramas)', fontsize=16, fontweight='bold')

for i, var in enumerate(variaveis_num):
    linha = i // 3
    coluna = i % 3

    sns.histplot(data=energy, x=var, kde=True, ax=axes[linha, coluna], color='steelblue')
    axes[linha, coluna].set_title(f'Distribuição de {var}')
    axes[linha, coluna].set_xlabel('')
    axes[linha, coluna].set_ylabel('Frequência')

plt.tight_layout()
plt.show()

