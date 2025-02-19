import matplotlib.pyplot as plt
from src.utils import load_grains_splited_datadet

#import linkage and dendogram
from scipy.cluster.hierarchy import linkage, dendrogram

X_train, samples, y_train, varieties = load_grains_splited_datadet()


# Calculate the linkage: mergings
mergings = mergings = linkage(samples, method='ward')

# Plot the dendrogram, using varieties as labels
plt.figure(figsize=(8, 5))
dendrogram(
    mergings,
    labels=varieties,  
    leaf_rotation=90,  
    leaf_font_size=10  
)
plt.title("Clusterização")
plt.xlabel("Variedades")
plt.ylabel("Distância")
plt.show()
plt.show()