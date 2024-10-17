# Import StandardScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from src.utils import load_wine_dataset
from sklearn.neighbors import KNeighborsClassifier
import numpy as np


wine = load_wine_dataset()

# Create the scaler
scaler = StandardScaler()

X = wine.drop(['Quality'],axis=1)

#X_norm = scaler.fit_transform(X)
X_norm = np.log(X)

y = wine['Quality'].values

print('variancia',X.var())
print('variancia do dataset normalizado',X_norm.var())

# Split the dataset and labels into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_norm, y, stratify=y, random_state=42)

knn = KNeighborsClassifier()

# Fit the k-nearest neighbors model to the training data
knn.fit(X_train,y_train)

# Score the model on the test data
print('score', knn.score(X_test, y_test))