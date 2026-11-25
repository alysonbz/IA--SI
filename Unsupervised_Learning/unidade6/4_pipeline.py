import pandas as pd

# Perform the necessary imports
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import k_means
from sklearn.preprocessing import StandardScaler
from src.utils import load_fish_dataset
from sklearn.pipeline import make_pipeline

samples_df = load_fish_dataset()
samples = samples_df.drop(['specie'],axis=1)
species = samples_df['specie'].values


# Create scaler: scaler
scaler =StandardScaler()

# Create KMeans instance: kmeans
kmeans = KMeans(n_clusters=4)

# Create pipeline: pipeline
pipeline = make_pipeline(scaler, kmeans)

# Fit the pipeline to samples
pipeline=pipeline.fit(samples)

# Calculate the cluster labels: labels
labels = pipeline.predict(samples)

# Create a DataFrame with labels and species as columns: df
df = ____

# Create crosstab: ct
ct = ____

# Display ct
print(ct)

