import pandas as pd
from src.utils import load_fish_dataset
from sklearn.cluster import KMeans

samples_df = load_fish_dataset()
samples = samples_df.drop(['specie'],axis=1)
specie = samples_df['specie'].values

# Create KMeans instance: kmeans with 4 custers
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)

# Use fit_predict to fit model and obtain cluster labels: labels
labels = kmeans.fit_predict(samples)

# Create a DataFrame with labels and varieties as columns: df
df = pd.DataFrame({'Cluster': labels, 'specie': specie})

# Create crosstab: ct
ct = pd.crosstab(df['Cluster'], df['specie'])
# Display ct
print(ct)