from src.utils import load_diabetes_clean_dataset
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler ,MinMaxScaler, MaxAbsScaler
#Import confusion matrix
from sklearn.metrics import classification_report, confusion_matriz

diabetes_df = load_diabetes_clean_dataset()
X = diabetes_df.drop(['diabetes'],axis=1)
scaler = MinMaxScaler()
x_nora = scaler.fit_transform(x)
y = diabetes_df['diabetes'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42,stratify=y)

knn = KNeighborsClassifier(n_neighbors=6)

# Fit the model to the training data
knn.fit(x_train,y_train)

# Predict the labels of the test data: y_pred
y_pred = knn.predict(x_test)

# Generate the confusion matrix and classification report
print(confusion_matriz(y_test, y_pred))
print(classification_report(y_test, y_pred))