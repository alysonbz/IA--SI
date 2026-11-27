import matplotlib.pyplot as plt
import pandas as pd
from src.utils import load_grains_dataset
from sklearn.cluster import KMeans

samples_df = load_grains_dataset()
samples = samples_df.drop(['variety', 'variety_number'], axis=1)

ks = range(1, 6)
inercias = []

for k in ks:
    # Create a KMeans instance with k clusters: model
    model = KMeans(n_clusters=k)

    # Fit model to samples
    model.fit(samples)

    # Append the inertia to the list of inertias
    inercias.append(model.inertia_)

# Plot ks vs inertias
plt.plot(ks, inercias, '-o')
plt.xlabel('number of clusters, k')
plt.ylabel('inertia')
plt.xticks(ks)
plt.show()