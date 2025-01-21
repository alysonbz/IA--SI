from src.utils import load_diabetes_clean_dataset
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

diabetes_df = load_diabetes_clean_dataset()

print(diabetes_df.isnull().sum())

#X = diabetes_df.drop(['diabetes'],axis=1)
#y = diabetes_df['diabetes'].values
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42,stratify=y)

#knn = KNeighborsClassifier(n_neighbors=6)

#knn.fit(X_train, y_train)

#y_pred = knn.predict(X_test)

#print(classification_report(y_test, y_pred))
#print(confusion_matrix(y_test, y_pred))