from src.utils import load_iris_dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

iris_df = load_iris_dataset()

le = LabelEncoder()

print(iris_df['class'].value_counts())

iris_df['class']= le.fit_transform(iris_df['class'])

print(iris_df['class'].value_counts())

y = iris_df['class'].values
x = iris_df.drop(['class'], axis=1)

X_train, X_teste, y_train, y_teste = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2)

knn= KNeighborsClassifier()

knn.fit(X_train,y_train)

print("Acurácia", knn.score(X_teste, y_teste))