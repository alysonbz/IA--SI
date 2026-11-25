import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from src.utils import load_fish_dataset
from sklearn.preprocessing import LabelEncoder

df = load_fish_dataset()
samples = df.drop(['specie'], axis=1)
scaler = StandardScaler()
scaled_samples = scaler.fit_transform(samples)


# Create a PCA model with components in adequate number: pca
pca = PCA(n_components=2)

# Fit the PCA instance to the scaled samples
pca.fit(scaled_samples)

# Transform the scaled samples: pca_features
pca_features = pca.transform(scaled_samples)

# Print the shape of pca_features
print(pca_features.shape)

# vizualize scatter plot with dimension reduced
species = LabelEncoder().fit_transform(df['specie'])
plt.scatter(pca_features[:, 0], pca_features[:, 1], c=species)
plt.xlabel('PCA feature 0')
plt.ylabel('PCA feature 1')
plt.show()
