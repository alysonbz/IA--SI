from _dados_ import *
import pandas as pd
from sklearn.preprocessing import StandardScaler

features_cluster = ["X1", "X2", "X3", "X4", "X5", "X7"]
dados_originais = energy[features_cluster].copy()

X_cluster = energy[features_cluster]

# Padronizando
scaler = StandardScaler()
X_padronizado = scaler.fit_transform(X_cluster)

X_padronizado = pd.DataFrame(X_padronizado, columns=features_cluster)

print("Dados originais (primeiras 3 linhas):")
print(dados_originais.head(3))
print("\nDados padronizados (primeiras 3 linhas):")
print(X_padronizado.head(3))