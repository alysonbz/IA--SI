# Perform the necessary imports
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from src.utils import load_fish_dataset

samples = load_fish_dataset()
samples = samples.drop(['specie'],axis=1)


# Create scaler: scaler
scaler = StandardScaler()

# Create a PCA instance: pca
pca = PCA()

# Create pipeline: pipeline
pipeline = make_pipeline(scaler, pca)

# Fit the pipeline to 'samples'
pipeline.fit(samples)

# Obter as variâncias explicadas
explained_variances = pca.explained_variance_ratio_

# Plot the explained variances
features = range(len(explained_variances))
plt.bar(features, explained_variances)
plt.xlabel('PCA feature')
plt.ylabel('variance')
plt.xticks(features)
plt.show()
