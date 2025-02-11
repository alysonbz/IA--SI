import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from src.utils import load_diabetes_clean_dataset

scaler = StandardScaler()
diabetes_df = load_diabetes_clean_dataset()
X = diabetes_df.drop(['diabetes'],axis=1)
X = scaler.fit_transform(X)
y = diabetes_df['diabetes'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#inicialize Logistic regression
knn = KNeighborsClassifier()

#inicialize kfold
kf = KFold(n_splits=5, shuffle=True, random_state=42)

#create the parameter space
params = {"n_neighbors":[5,6,7,8,9,10],
         "metric": ["minkowski", "euclidean", "manhattan"]}

# Instantiate the RandomizedSearchCV object
knn_cv = RandomizedSearchCV(knn, params, cv=kf)

# Fit the data to the model
knn_cv.fit(X_train, y_train)

# Print the tuned parameters and score
print("KNN Parameters: {}".format(knn_cv.best_params_))
print("KNN Best Accuracy Score: {}".format(knn_cv.best_score_))