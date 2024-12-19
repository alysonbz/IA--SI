from src.utils import load_iris_dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

iris_df = load_iris_dataset()

le = LabelEncoder()

#Exibindo a quantidaade de elementos de cada classe
print(iris_df['class'].value_counts())

#trocando nomes por labels
iris_df['class']=le.fit_transform(iris_df['class'])

#Exibindo a quantidade de elementos de cada classe numericamente
print(iris_df['class'].value_counts())

#separação em treino e teste
y = iris_df['class'].values
X = iris_df.drop(['class'],axis=1)

X_train, X_teste, y_train, y_teste = train_test_split(X,y,test_size=0.2,stratify=y,random_state=2)

knn = KNeighborsClassifier()

knn.fit(X_train,y_train)

print("Acurácia ",knn.score(X_teste,y_teste))