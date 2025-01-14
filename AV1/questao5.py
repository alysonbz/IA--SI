import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

dataset_path = 'bodyfat.csv'

df = pd.read_csv(dataset_path)

print(df.head())

X = df.drop("BodyFat", axis=1) 
y = df["BodyFat"]  

print(df.isnull().sum()) 
df.dropna(inplace=True)

X = df.drop("BodyFat", axis=1)
y = df["BodyFat"]

correlation_matrix = df.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Matriz de Correlação entre as variáveis")
plt.show()

correlations = correlation_matrix["BodyFat"].sort_values(ascending=False)
print(correlations)

top_features = correlations[1:6]  
plt.figure(figsize=(10, 6))
top_features.plot(kind='bar', color='skyblue')
plt.title('Correlação das variáveis mais relevantes com BodyFat')
plt.ylabel('Correlação')
plt.xlabel('Variáveis')
plt.show()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)