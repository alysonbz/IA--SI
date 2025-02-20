import pandas as pd

# Importar os módulos necessários
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline

from src.utils import load_fish_dataset

samples_df = load_fish_dataset()
samples = samples_df.drop(['specie'],axis=1)
species = samples_df['specie'].values

# Create scaler: scaler
scaler = StandardScaler()

# Create KMeans instance: kmeans
kmeans = KMeans(n_clusters=4, random_state=42)

# Create pipeline: pipeline
pipeline = make_pipeline(scaler, kmeans)

# Fit the pipeline to samples
pipeline.fit(samples)

# Calculate the cluster labels: labels
labels = pipeline.predict(samples)

# Create a DataFrame with labels and species as columns: df
df = pd.DataFrame({'Labels': labels, 'Species': species})

# Create crosstab: ct
ct = pd.crosstab(df['Labels'], df['Species'])

# Display ct
print(ct)

