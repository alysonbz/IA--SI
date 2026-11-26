# QUESTÃO 6 — INTERPRETAÇÃO DOS CLUSTERS
# Dataset: Energy Efficiency

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from _dados_ import *
from _pre_process_ import X_padronizado, dados_originais

K_FINAL = 3

kmeans_final = KMeans(n_clusters=K_FINAL, random_state=42, n_init=10)
kmeans_final.fit(X_padronizado)

dados_cluster = dados_originais.copy()
dados_cluster['cluster'] = kmeans_final.labels_

# (Opcional) salva o CSV para reutilizar depois
dados_cluster.to_csv("dados_com_clusters.csv", index=False)

FEATURES = ["X1", "X2", "X3", "X4", "X5", "X7"]

sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

print("=" * 70)
print("INTERPRETAÇÃO DOS CLUSTERS — ENERGY EFFICIENCY")
print("=" * 70)
print(f"Total de amostras: {len(dados_cluster)}")
print(f"Clusters identificados: {sorted(dados_cluster['cluster'].unique())}")
print("=" * 70)
# Features usadas no agrupamento
FEATURES = ["X1", "X2", "X3", "X4", "X5", "X7"]

sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

print("=" * 70)
print("INTERPRETAÇÃO DOS CLUSTERS — ENERGY EFFICIENCY")
print("=" * 70)
print(f"Total de amostras: {len(dados_cluster)}")
print(f"Clusters identificados: {sorted(dados_cluster['cluster'].unique())}")
print("=" * 70)

# -------------------------------------------------------------
# 6.a) Médias e medianas das principais características por cluster
# -------------------------------------------------------------
medias = dados_cluster.groupby('cluster')[FEATURES].mean().round(3)
medianas = dados_cluster.groupby('cluster')[FEATURES].median().round(3)

print("\n" + "=" * 70)
print("6.a) MÉDIAS POR CLUSTER")
print("=" * 70)
print(medias.to_string())

print("\n" + "=" * 70)
print("6.a) MEDIANAS POR CLUSTER")
print("=" * 70)
print(medianas.to_string())

# -------------------------------------------------------------
# 6.b) Tabela comparativa (média + mediana)
# -------------------------------------------------------------
tabela_comparativa = pd.concat(
    [medias, medianas],
    axis=1,
    keys=['Média', 'Mediana']
)

print("\n" + "=" * 70)
print("6.b) TABELA COMPARATIVA (MÉDIA + MEDIANA)")
print("=" * 70)
print(tabela_comparativa.to_string())

# Salva a tabela em CSV para o relatório
tabela_comparativa.to_csv("tabela_comparativa_clusters.csv")
print("\n[OK] Tabela salva em 'tabela_comparativa_clusters.csv'")

# -------------------------------------------------------------
# 6.c) Visualização 1 — Boxplots por cluster
# -------------------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Distribuição das Features por Cluster (Boxplots)',
             fontsize=16, fontweight='bold')

for i, var in enumerate(FEATURES):
    linha, coluna = i // 3, i % 3
    sns.boxplot(
        data=dados_cluster, x='cluster', y=var,
        ax=axes[linha, coluna], palette='Set2'
    )
    axes[linha, coluna].set_title(f'{var} por Cluster')
    axes[linha, coluna].set_xlabel('Cluster')
    axes[linha, coluna].set_ylabel(var)

plt.tight_layout()
plt.show()

# -------------------------------------------------------------
# 6.c) Visualização 2 — Scatter 2D com cores por cluster
# Usa as duas features mais discriminantes (X2 e X5)
# -------------------------------------------------------------
plt.figure(figsize=(10, 7))
sns.scatterplot(
    data=dados_cluster,
    x='X2', y='X5',
    hue='cluster',
    palette='Set1',
    alpha=0.7, s=60
)
plt.title('Clusters: Área de Superfície (X2) vs Altura (X5)')
plt.xlabel('Área de Superfície (X2)')
plt.ylabel('Altura (X5)')
plt.legend(title='Cluster')
plt.tight_layout()
plt.show()

# -------------------------------------------------------------
# 6.c) Visualização 3 (extra) — Parallel Coordinates padronizado
# Mostra o "perfil" de cada cluster em escala padronizada
# -------------------------------------------------------------
scaler = StandardScaler()
dados_norm = pd.DataFrame(
    scaler.fit_transform(dados_cluster[FEATURES]),
    columns=FEATURES
)
dados_norm['cluster'] = dados_cluster['cluster'].values

plt.figure(figsize=(12, 6))
pd.plotting.parallel_coordinates(
    dados_norm, 'cluster',
    colormap='Set1', alpha=0.5
)
plt.title('Perfil Padronizado das Features por Cluster')
plt.ylabel('Valor padronizado (z-score)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# -------------------------------------------------------------
# 6.d) Principais características que diferenciam os grupos
# -------------------------------------------------------------
variancia_entre = medias.var().sort_values(ascending=False)

print("\n" + "=" * 70)
print("6.d) FEATURES MAIS DISCRIMINANTES (variância das médias entre clusters)")
print("=" * 70)
print(variancia_entre.to_string())

# Normaliza para ter uma noção relativa (0 a 1)
var_norm = (variancia_entre / variancia_entre.max()).round(3)
print("\nImportância relativa (0 a 1):")
print(var_norm.to_string())

top_features = variancia_entre.head(2).index.tolist()
print(f"\n>> Features que mais diferenciam os clusters: {top_features}")

# -------------------------------------------------------------
# 6.e) Descrição interpretativa de cada cluster
# As descrições abaixo são um TEMPLATE — ajuste conforme seus números
# -------------------------------------------------------------
print("\n" + "=" * 70)
print("6.e) DESCRIÇÃO INTERPRETATIVA DOS CLUSTERS")
print("=" * 70)

# Gera descrição automática baseada nas médias (ranking dentro de cada feature)
def descrever_cluster(cluster_id):
    m = medias.loc[cluster_id]
    descricoes = []

    # Altura (X5)
    if m['X5'] >= medias['X5'].quantile(0.66):
        descricoes.append("edifícios ALTOS")
    elif m['X5'] <= medias['X5'].quantile(0.33):
        descricoes.append("edifícios BAIXOS")
    else:
        descricoes.append("edifícios de altura MÉDIA")

    # Área de superfície (X2)
    if m['X2'] >= medias['X2'].quantile(0.66):
        descricoes.append("grande área de superfície")
    elif m['X2'] <= medias['X2'].quantile(0.33):
        descricoes.append("pequena área de superfície")
    else:
        descricoes.append("área de superfície intermediária")

    # Compacidade (X1)
    if m['X1'] >= medias['X1'].quantile(0.66):
        descricoes.append("formato COMPACTO")
    elif m['X1'] <= medias['X1'].quantile(0.33):
        descricoes.append("formato alongado")
    else:
        descricoes.append("compacidade média")

    # Área envidraçada (X7)
    if m['X7'] >= medias['X7'].quantile(0.66):
        descricoes.append("muita área envidraçada")
    elif m['X7'] <= medias['X7'].quantile(0.33):
        descricoes.append("pouca área envidraçada")
    else:
        descricoes.append("área envidraçada moderada")

    return ", ".join(descricoes)

for c in sorted(dados_cluster['cluster'].unique()):
    n = (dados_cluster['cluster'] == c).sum()
    pct = n / len(dados_cluster) * 100
    desc = descrever_cluster(c)
    print(f"\nCluster {c}  ({n} amostras — {pct:.1f}%)")
    print(f"  Perfil: {desc}")
    print(f"  Médias: X1={medias.loc[c,'X1']:.2f} | X2={medias.loc[c,'X2']:.1f} | "
          f"X3={medias.loc[c,'X3']:.1f} | X4={medias.loc[c,'X4']:.1f} | "
          f"X5={medias.loc[c,'X5']:.2f} | X7={medias.loc[c,'X7']:.2f}")

# -------------------------------------------------------------
# 6.f) Discussão: padrões relevantes e limitações
# -------------------------------------------------------------
print("\n" + "=" * 70)
print("6.f) DISCUSSÃO — PADRÕES RELEVANTES E LIMITAÇÕES")
print("=" * 70)


# -------------------------------------------------------------
# Extra: relação entre clusters e os alvos Y1/Y2
# -------------------------------------------------------------
if 'Y1' in dados_cluster.columns and 'Y2' in dados_cluster.columns:
    print("=" * 70)
    print("EXTRA) MÉDIAS DOS ALVOS Y1 e Y2 POR CLUSTER")
    print("=" * 70)
    alvos = dados_cluster.groupby('cluster')[['Y1', 'Y2']].mean().round(2)
    print(alvos.to_string())

    plt.figure(figsize=(10, 5))
    alvos.plot(kind='bar', ax=plt.gca(), color=['#1f77b4', '#ff7f0e'])
    plt.title('Carga de Aquecimento (Y1) e Resfriamento (Y2) por Cluster')
    plt.xlabel('Cluster')
    plt.ylabel('Carga (kWh/m²)')
    plt.xticks(rotation=0)
    plt.legend(['Y1 — Aquecimento', 'Y2 — Resfriamento'])
    plt.tight_layout()
    plt.show()
else:
    print("\n[AVISO] Colunas Y1/Y2 não encontradas em dados_cluster.")
    print("Use dados_originais.merge(...) para adicioná-las caso queira.")

print("\n[FIM] Questão 6 concluída.")