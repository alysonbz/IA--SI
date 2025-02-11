import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from src.utils import load_fish_dataset
from sklearn.cluster import KMeans

samples_df = load_fish_dataset()
samples = samples_df.drop(['specie'],axis=1)
varieties = samples_df['specie'].values

# Create KMeans instance: kmeans with 4 custers
kmeans = KMeans(n_clusters=4)
scaler = StandardScaler()
pipeline = make_pipeline(scaler, kmeans)
# Use fit_predict to fit model and obtain cluster labels: labels
labels = pipeline.fit_predict(samples)

# Create a DataFrame with labels and varieties as columns: df
df = pd.DataFrame({'labels': labels, 'specie': varieties})

# Create crosstab: ct
ct = pd.crosstab(df['labels'], df['specie'])
# Display ct
print(ct)