import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.metrics import silhouette_score

df = pd.read_csv(r"C:\Users\Administrator\IA--SI\AV2\healthcare-dataset-stroke-data.csv")

X = df.drop(columns=["stroke"])
X = X.apply(lambda col: col.fillna(col.mean()) if col.dtype in ['int64', 'float64'] else col)

X = pd.get_dummies(X, drop_first=True)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

lasso = Lasso(alpha=0.01)
lasso.fit(X_scaled, np.zeros(X_scaled.shape[0]))

coeficientes = np.abs(lasso.coef_)
atributos = np.array(X.columns)
indices_top2 = np.argsort(coeficientes)[-2:]
atributos_top2 = atributos[indices_top2]

print(f"Os dois atributos mais importantes selecionados pelo Lasso: {atributos_top2}")

X_selected = X[atributos_top2]

inertia = []
silhouette_scores = []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_selected)
    inertia.append(kmeans.inertia_)

    if k > 1:
        silhouette_scores.append(silhouette_score(X_selected, clusters))
    else:
        silhouette_scores.append(0)

best_k_silhouette = K_range[np.argmax(silhouette_scores)]
print(f"Melhor número de clusters baseado na silhueta: {best_k_silhouette}")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(K_range, inertia, 'o--b')
axes[0].set_title("Método do Cotovelo")
axes[0].set_xlabel("Número de Clusters (k)")
axes[0].set_ylabel("Inércia")

axes[1].plot(K_range, silhouette_scores, 'o--r')
axes[1].set_title("Análise de Silhueta")
axes[1].set_xlabel("Número de Clusters (k)")
axes[1].set_ylabel("Coeficiente de Silhueta")
plt.show()
