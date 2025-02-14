import pandas as pd

# Perform the necessary imports
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.utils import load_fish_dataset

samples_df = load_fish_dataset()
samples = samples_df.drop(['specie'],axis=1)
species = samples_df['specie'].values

# Create scaler: scaler
scaler = StandardScaler()

# Create KMeans instance: kmeans
kmeans = KMeans(n_clusters=4)

# Create pipeline: pipeline
pipeline = make_pipeline(scaler, kmeans)

# Fit the pipeline to samples
pipeline.fit(samples)

# Calculate the cluster labels: labels
labels = pipeline.fit_predict(samples)

# Create a DataFrame with labels and species as columns: df
df = pd.DataFrame({'labels': labels, 'specie': species})

# Create crosstab: ct
ct = pd.crosstab(df['labels'], df['specie'])

# Display ct
print(ct)

