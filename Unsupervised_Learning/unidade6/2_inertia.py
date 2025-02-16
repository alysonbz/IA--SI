import matplotlib.pyplot as plt
import pandas as pd
from src.utils import load_grains_dataset
from sklearn.cluster import KMeans

# Load the dataset
samples_df = load_grains_dataset()
samples = samples_df.drop(['variety', 'variety_number'], axis=1)

# Set the range of k values (number of clusters)
ks = range(1, 6)
inertias = []

for k in ks:
    # Create a KMeans instance with k clusters
    model = KMeans(n_clusters=k, random_state=42)

    # Fit model to samples
    model.fit(samples)

    # Append the inertia (sum of squared distances) to the list of inertias
    inertias.append(model.inertia_)

# Plot ks vs inertias
plt.plot(ks, inertias, '-o')
plt.xlabel('number of clusters, k')
plt.ylabel('inertia')
plt.xticks(ks)
plt.show()
