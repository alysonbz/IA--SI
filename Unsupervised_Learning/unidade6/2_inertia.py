import matplotlib.pyplot as plt
import pandas as pd
from src.utils import load_grains_dataset
from sklearn.cluster import KMeans

# Carregar dataset
samples_df = load_grains_dataset()
samples = samples_df.drop(['variety', 'variety_number'], axis=1)

# Definir intervalo de clusters a serem testados
ks = range(1, 6)
inertias = []

# Loop para testar diferentes valores de k
for k in ks:
    # Criar um modelo KMeans com k clusters
    model = KMeans(n_clusters=k, random_state=42)

    # Ajustar modelo aos dados
    model.fit(samples)

    # Salvar inércia do modelo atual
    inertias.append(model.inertia_)

# Plotar gráfico de Elbow Method
plt.plot(ks, inertias, '-o')
plt.xlabel('Número de clusters (k)')
plt.ylabel('Inércia')
plt.xticks(ks)
plt.show()
