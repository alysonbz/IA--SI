# QUESTÃO 5 — APLICAÇÃO DO K-MEANS
# Dataset: Energy Efficiency

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from _pre_process_ import X_padronizado, dados_originais

sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# 5.a) Executar K-Means para K de 2 a 10
# 5.b) Calcular a inércia para cada K
# 5.d) Calcular o Silhouette Score para cada K

K_range = range(2, 11)
inercias = []
silhouette_scores = []
modelos = {}

print("=" * 60)
print("TREINAMENTO K-MEANS PARA K = 2..10")
print("=" * 60)
print(f"{'K':>3} | {'Inércia':>12} | {'Silhouette':>12}")
print("-" * 35)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_padronizado)
    inercia = kmeans.inertia_
    sil = silhouette_score(X_padronizado, kmeans.labels_)

    inercias.append(inercia)
    silhouette_scores.append(sil)
    modelos[k] = kmeans

    print(f"{k:>3} | {inercia:>12.2f} | {sil:>12.4f}")

# -------------------------------------------------------------
# 5.c) Gráfico do Método do Cotovelo
# -------------------------------------------------------------
plt.figure(figsize=(8, 5))
plt.plot(list(K_range), inercias, 'bo-', linewidth=2, markersize=8)
plt.xlabel('Número de Clusters (K)')
plt.ylabel('Inércia')
plt.title('Método do Cotovelo (Elbow Method)')
plt.xticks(list(K_range))
plt.grid(True)
plt.tight_layout()
plt.show()

# -------------------------------------------------------------
# 5.e) Seleção do K — gráfico combinado Inércia x Silhouette
# -------------------------------------------------------------
fig, ax1 = plt.subplots(figsize=(10, 5))

ax1.plot(list(K_range), inercias, 'bo-', label='Inércia')
ax1.set_xlabel('K')
ax1.set_ylabel('Inércia', color='b')
ax1.tick_params(axis='y', labelcolor='b')

ax2 = ax1.twinx()
ax2.plot(list(K_range), silhouette_scores, 'rs-', label='Silhouette')
ax2.set_ylabel('Silhouette Score', color='r')
ax2.tick_params(axis='y', labelcolor='r')

plt.title('Inércia vs Silhouette por K')
fig.tight_layout()
plt.show()

# Escolha do K final

K_FINAL = 3

print("\n" + "=" * 60)
print(f"K ESCOLHIDO: {K_FINAL}")
print(f"  Inércia   = {inercias[K_FINAL - 2]:.2f}")
print(f"  Silhouette= {silhouette_scores[K_FINAL - 2]:.4f}")
print("Justificativa: ponto do cotovelo + Silhouette estável/alto +")
print("interpretabilidade no contexto de eficiência energética.")
print("=" * 60)

# -------------------------------------------------------------
# 5.f) Executar o agrupamento final com K escolhido
# -------------------------------------------------------------
kmeans_final = KMeans(n_clusters=K_FINAL, random_state=42, n_init=10)
kmeans_final.fit(X_padronizado)

# Adiciona rótulos ao dataframe original (preserva unidades originais)
dados_cluster = dados_originais.copy()
dados_cluster['cluster'] = kmeans_final.labels_

print("\nPrimeiras linhas com rótulo de cluster:")
print(dados_cluster.head())

# -------------------------------------------------------------
# 5.g) Quantidade e percentual de amostras por cluster
# -------------------------------------------------------------
contagem = dados_cluster['cluster'].value_counts().sort_index()
percentual = (contagem / len(dados_cluster) * 100).round(2)

print("\n" + "=" * 60)
print("DISTRIBUIÇÃO DOS CLUSTERS")
print("=" * 60)
print(f"{'Cluster':>7} | {'Amostras':>9} | {'Percentual':>10}")
print("-" * 35)
for c in contagem.index:
    print(f"{c:>7} | {contagem[c]:>9} | {percentual[c]:>9.2f}%")
print("-" * 35)
print(f"{'Total':>7} | {contagem.sum():>9} | {100.00:>9.2f}%")

plt.figure(figsize=(10, 7))
sns.scatterplot(
    data=dados_cluster,
    x='X2', y='X5',
    hue='cluster',
    palette='Set1',
    alpha=0.7, s=60
)
plt.title(f'Clusters K-Means (K={K_FINAL}) — X2 vs X5')
plt.xlabel('Área de Superfície (X2)')
plt.ylabel('Altura (X5)')
plt.legend(title='Cluster')
plt.tight_layout()
plt.show()