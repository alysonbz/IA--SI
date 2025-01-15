from pandas.io.xml import preprocess_data

from src.utils import load_churn_dataset
import numpy as np

# Import KNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier

churn_df = load_churn_dataset()

# Create arrays for the features and the target variable
y = churn_df["churn"].values
<<<<<<< HEAD
X = churn_df[["account_length", "number_customer_service_calls"]].values

# Create a KNN classifier with 6 neighbors
knn = KNeighborsClassifier(n_neighbors=6)
=======
X = churn_df[["account_lenght", "number_customer_service_calls"]].values

# Create a KNN classifier with 6 neighbors
knn = KNeighborsClassifier(n_neighbors=5)
>>>>>>> 1239a00c96cd4d3adea696c64633d52b04d5adf1

# Fit the classifier to the data
knn.fit(X, y)

X_test = np.array([[30.0, 17.5],
                  [107.0, 24.1],
                  [213.0, 10.9]])

# Predict the labels for the X_teste
y_pred = knn.predict(X_test)

# Print the predictions for X_test
print("Predictions: {}".format(y_pred))