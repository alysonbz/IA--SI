import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def load_combined_data():
    return pd.read_csv('credit_risk_data.csv')

credit_risk_data = load_combined_data()

# Definição X e y
X = credit_risk_data.drop(['label'], axis=1).values  #Features
y = credit_risk_data['label'].values  #Target

#Divisão do dataset em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#Normalização de Média Zero e Variância Unitária
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Melhor k
train_accuracies = {}
test_accuracies = {}
neighbors = np.arange(1, 12)

for neighbor in neighbors:
    knn = KNeighborsClassifier(n_neighbors=neighbor)
    knn.fit(X_train_scaled, y_train)

    #Acuracia
    train_accuracies[neighbor] = knn.score(X_train_scaled, y_train)
    test_accuracies[neighbor] = knn.score(X_test_scaled, y_test)

#Resultado Acuracia
print("Acurácia no Treino: ", train_accuracies)
print("Acurácia no Teste: ", test_accuracies)

best_k = max(test_accuracies, key=test_accuracies.get)
best_accuracy = test_accuracies[best_k]
print(f'O melhor valor de K é: {best_k} com uma acurácia de: {best_accuracy:.2f}')

#Plotando as acuracias
plt.figure(figsize=(10, 6))
plt.title("KNN: Variação do Número de Vizinhos")
plt.plot(neighbors, train_accuracies.values(), label="Acurácia de Treinamento", marker='o')
plt.plot(neighbors, test_accuracies.values(), label="Acurácia de Teste", marker='o')
plt.legend()
plt.xlabel("Número de Vizinhos (k)")
plt.ylabel("Acurácia")
plt.xticks(neighbors)
plt.grid()
plt.show()