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


# Função para carregar e processar em chunks
def load_and_process_in_chunks(file_path, chunk_size=100000):
    chunks = pd.read_csv(file_path, chunksize=chunk_size)
    df_list = []

    for chunk in chunks:
        # Limpeza de dados: Remover NaNs e valores infinitos
        chunk = chunk.fillna(0)
        chunk.replace([np.inf, -np.inf], 0, inplace=True)
        chunk = chunk.apply(pd.to_numeric, errors='coerce').fillna(0)

        # Selecionar apenas as colunas necessárias
        colunas_relevantes = ['alpha', 'delta', 'u', 'g', 'r', 'i', 'z', 'redshift', 'plate', 'MJD', 'class']
        chunk = chunk[colunas_relevantes]

        # Codificar a variável target
        chunk['class'] = LabelEncoder().fit_transform(chunk['class'])

        df_list.append(chunk)

    return pd.concat(df_list, axis=0)


# Caminho do arquivo
file_path = r"C:\Users\bende\av1\IA--SI\AV2\star_classification.csv"

# Carregar e processar dados em chunks
df = load_and_process_in_chunks(file_path)

# Análise exploratória
sns.countplot(x=df['class'])
plt.title('Distribuição das Classes')
plt.xticks(ticks=range(len(df['class'].unique())), labels=np.unique(df['class']))
plt.show()

# Preparar dados para modelagem
X, y = df.drop('class', axis=1), df['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Reduzir a dimensionalidade com PCA
pca = PCA(n_components=0.95)  # Conservar 95% da variação
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Configurar e executar GridSearchCV (usando uma amostra para acelerar o processo)
param_grid = {
    'n_neighbors': range(1, 30),
    'metric': ['euclidean', 'manhattan', 'minkowski'],
    'weights': ['uniform', 'distance']
}
grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, scoring='accuracy', cv=5, n_jobs=-1)

# Usar apenas uma amostra do treinamento para GridSearch (reduzir o tempo)
X_train_sampled, _, y_train_sampled, _ = train_test_split(X_train_pca, y_train, test_size=0.9, random_state=42)

grid_search.fit(X_train_sampled, y_train_sampled)

# Avaliação final
y_pred = grid_search.best_estimator_.predict(X_test_pca)
print(f"\nMelhores parâmetros: {grid_search.best_params_}")
print(f"Acurácia no teste: {accuracy_score(y_test, y_pred):.4f}")
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Salvar dataset processado
df.to_csv(r"C:\Users\bende\av1\IA--SI\AV2\star_classification_processed.csv", index=False)
