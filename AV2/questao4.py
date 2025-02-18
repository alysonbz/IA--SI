import pandas as pd
from sklearn.cluster import KMeans

df = pd.read_csv('sabores_de_cacau_ajustado.csv')
samples = df.drop(columns=['variety', 'variety_number'])
varieties = df['variety'].values
model = KMeans(n_clusters=2, random_state=42, n_init=10)
labels = model.fit_predict(samples)
df_clusters = pd.DataFrame({'labels': labels, 'varieties': varieties})
ct = pd.crosstab(df_clusters['labels'], df_clusters['varieties'])
print(ct)