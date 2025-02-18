import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from AV1.questao6 import model, scaler
from src.utils import load_fish_dataset
from sklearn.cluster import KMeans

samples_df = load_fish_dataset()
samples = samples_df.drop(['specie'],axis=1)
varieties = samples_df['specie'].values

# Create KMeans instance: kmeans with 4 custers
model = KMeans(n_clusters=4)
scaler = StandardScaler()
pipeline = make_pipeline(scaler, model)

# Use fit_predict to fit model and obtain cluster labels: labels
label = model.fit(samples)

# Create a DataFrame with labels and varieties as columns: df
df = pd.DataFrame({'label': label, 'specie': varieties})

# Create crosstab: ct
ct = pd.crosstab(df['label'], df['specie'])
# Display ct
print(ct)