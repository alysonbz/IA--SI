# Import TSNE
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from src.utils import load_movements_price_dataset
from sklearn.preprocessing import normalize

# Create a TSNE instance: model
model = TSNE(n_components=2, learning_rate=200, random_state=42)


movements_df = load_movements_price_dataset()
movements = movements_df.drop(['company'],axis=1)
companies = movements_df['company'].values
normalized_movements = normalize(movements)

# Normalize the movements
normalized_movements = normalize(movements)

# Apply fit_transform to normalized_movements: tsne_features
tsne_features = model.fit_transform(normalized_movements)

# Select the 0th feature: xs
xs = tsne_features[:, 0]

# Select the 1th feature: ys
ys = tsne_features[:,1]

# Scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(xs, ys, alpha=0.7)

# Annotate the points
for x, y, company in zip(xs, ys, companies):
    plt.annotate(company, (x, y), fontsize=5, alpha=0.75)
plt.show()
