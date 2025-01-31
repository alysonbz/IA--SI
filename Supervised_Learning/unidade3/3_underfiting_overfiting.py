from src.utils import load_churn_dataset
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

churn_df = load_churn_dataset()
X = churn_df[["account_length", "total_day_charge", "total_eve_charge", "total_night_charge", "total_intl_charge", "number_customer_service_calls"]].values
y = churn_df["churn"].values

# Divisão em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Criação de vizinhos
neighbors = np.arange(1, 21)
train_accuracies = {}
test_accuracies = {}

for neighbor in neighbors:
    # Configuração do classificador KNN
    knn = KNeighborsClassifier(n_neighbors=neighbor)

    # Ajuste do modelo
    knn.fit(X_train, y_train)

    # Computação da acurácia
    train_accuracies[neighbor] = knn.score(X_train, y_train)
    test_accuracies[neighbor] = knn.score(X_test, y_test)

print("Acurácia no treino: ", train_accuracies, '\n', "Acurácia no teste: ", test_accuracies)

# Adicionar um título
plt.title("Acurácia do KNN para diferentes números de vizinhos")

# Plotando as acurácias de treino
plt.plot(neighbors, list(train_accuracies.values()), label="Acurácia de Treino")

# Plotando as acurácias de teste
plt.plot(neighbors, list(test_accuracies.values()), label="Acurácia de Teste")

plt.legend()
plt.xlabel("Número de Vizinhos")
plt.ylabel("Acurácia")

# Exibindo o gráfico
plt.show()
