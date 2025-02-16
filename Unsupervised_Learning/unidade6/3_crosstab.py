import pandas as pd
from src.utils import load_grains_dataset
from sklearn.cluster import KMeans

# Load the dataset
samples_df = load_grains_dataset()
samples = samples_df.drop(['variety', 'variety_number'], axis=1)
varieties = samples_df['variety'].values

# Create a KMeans model with 3 clusters
model = KMeans(n_clusters=3)

# Use fit_predict to fit the model and obtain cluster labels
labels = model.fit_predict(samples)

# Create a DataFrame with labels and varieties as columns
df = pd.DataFrame({'labels': labels, 'varieties': varieties})

# Create a crosstab of labels vs varieties
ct = pd.crosstab(df['labels'], df['varieties'])

# Display the crosstab
print(ct)
