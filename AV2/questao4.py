import pandas as pd
from sklearn.cluster import KMeans
from AV2.src.utills import diabetes_dataset
from sklearn.preprocessing import StandardScaler

diabetes = diabetes_dataset()

samples = diabetes.drop(columns=['Outcome'])
varieties = diabetes['Outcome']

scaler = StandardScaler()
samples_scaled = scaler.fit_transform(samples)

model = KMeans(n_clusters=2)

labels = model.fit_predict(samples)

new_diabetes = pd.DataFrame({'labels': labels, 'varieties': varieties})

ct = pd.crosstab(new_diabetes['labels'], new_diabetes['varieties'])

print(ct)