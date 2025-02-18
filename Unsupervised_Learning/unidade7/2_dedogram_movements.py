import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.preprocessing import normalize
from src.utils import load_movements_price_dataset

movements_df = load_movements_price_dataset()
movements = movements_df.drop(['company'],axis=1)
companies = movements_df['company'].values

# Normalize the movements: normalized_movements
normalized_movements = normalize(movements, axis=0)

# Calculate the linkage: mergings
mergings = linkage(normalized_movements, method='ward')

# Plot the dendrogram
dendrogram(mergings, labels=companies)
plt.show()