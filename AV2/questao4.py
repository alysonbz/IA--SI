import pandas as pd
from sklearn.cluster import KMeans

df = pd.read_csv(r'C:\Users\adrie\OneDrive\Documentos\IA-Si\IA--SI\AV2\Cancer_Data.csv')

df = df.drop(columns=["id", "Unnamed: 32"], errors='ignore')

df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})

# Separar features e variável alvo
samples = df.drop(columns=["diagnosis"])
varieties = df["diagnosis"]

model = KMeans(n_clusters=2)

labels = model.fit_predict(samples)

new_df = pd.DataFrame({'labels': labels, 'varieties': varieties})

ct = pd.crosstab(new_df['labels'], new_df['varieties'])

print(ct)
