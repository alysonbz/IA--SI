from src.utils import load_wine_dataset
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import numpy as np
wine = load_wine_dataset()

print(wine.info())

# Split the dataset into training and test sets
X = wine.drop(['Quality'],axis=1)
#X= np.log(X)
y = wine['Quality'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

knn = KNeighborsClassifier(n_neighbors=5)

# Fit the knn model to the training data
knn.fit(X_train, y_train)

pred = knn.predict(X_test)

# Score the model on the test data
print(knn.score(X_test,y_test))

print("knn result: ",pred, "label: ",y_test)