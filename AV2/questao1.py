import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer


def load_and_process_in_chunks(file_path, chunk_size=100000):
    chunks = pd.read_csv(file_path, chunksize=chunk_size)
    df_list = []

    for chunk in chunks:
        chunk = chunk.fillna(0)
        chunk.replace([np.inf, -np.inf], 0, inplace=True)
        chunk = chunk.apply(pd.to_numeric, errors='coerce').fillna(0)

        colunas_relevantes = ['alpha', 'delta', 'u', 'g', 'r', 'i', 'z', 'redshift', 'plate', 'MJD', 'class']
        chunk = chunk[colunas_relevantes]

        chunk['class'] = LabelEncoder().fit_transform(chunk['class'])

        df_list.append(chunk)

    return pd.concat(df_list, axis=0)


file_path = r"C:\Users\bende\av1\IA--SI\AV2\star_classification.csv"

df = load_and_process_in_chunks(file_path)

sns.countplot(x=df['class'])
plt.title('Distribuição das Classes')
plt.xticks(ticks=range(len(df['class'].unique())), labels=np.unique(df['class']))
plt.show()

X, y = df.drop('class', axis=1), df['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

pca = PCA(n_components=0.95)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

param_grid = {
    'n_neighbors': range(1, 30),
    'metric': ['euclidean', 'manhattan', 'minkowski'],
    'weights': ['uniform', 'distance']
}
grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, scoring='accuracy', cv=5, n_jobs=-1)

X_train_sampled, _, y_train_sampled, _ = train_test_split(X_train_pca, y_train, test_size=0.9, random_state=42)

grid_search.fit(X_train_sampled, y_train_sampled)

y_pred = grid_search.best_estimator_.predict(X_test_pca)
print(f"\nMelhores parâmetros: {grid_search.best_params_}")
print(f"Acurácia no teste: {accuracy_score(y_test, y_pred):.4f}")
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

df.to_csv(r"C:\Users\bende\av1\IA--SI\AV2\star_classification_processed.csv", index=False)
