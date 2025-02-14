import matplotlib.pyplot as plt
from nltk.sem.chat80 import country
from src.utils import load_grains_splited_datadet
from scipy.cluster.hierarchy import linkage, dendrogram

X_train, samples, y_train, varieties = load_grains_splited_datadet()

# Calculate the linkage: mergings
mergings = linkage(samples, method='complete')

# Plot the dendrogram, using varieties as labels
dendrogram(mergings,
           labels=varieties,
           leaf_rotation=90,
           leaf_font_size=6,
)
plt.show()