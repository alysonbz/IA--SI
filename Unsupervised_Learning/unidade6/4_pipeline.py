import pandas as pd
from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

from src.utils import load_fish_dataset

samples_df = load_fish_dataset()
samples = samples_df.drop(['specie'],axis=1)
species = samples_df['specie'].values


# Create scaler: scaler
scaler = StandardScaler()

# Create KMeans instance: kmeans
model = KMeans(n_clusters=4)

# Create pipeline: pipeline
pipeline = make_pipeline(scaler, model)

# Fit the pipeline to samples
pipeline.fit(samples)

# Calculate the cluster labels: labels
labels = model.fit_predict(samples)

# Create a DataFrame with labels and species as columns: df
df = pd.DataFrame({'labels': labels, 'species': species})

# Create crosstab: ct
ct = pd.crosstab(df['labels'], df['species'])

# Display ct
print(ct)

