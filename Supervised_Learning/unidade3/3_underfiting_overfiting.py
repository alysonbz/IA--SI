from src.utils import load_churn_dataset
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

churn_df = load_churn_dataset()

X = churn_df[["account_length",  "total_day_charge", "total_eve_charge", "total_night_charge", "total_intl_charge", "number_customer_service_calls"]].values
y = churn_df["churn"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

neighbors = np.arange(1, 21)
train_accuracies = {}
test_accuracies = {}

for neighbor in neighbors:
    knn = KNeighborsClassifier(n_neighbors=neighbor)

    knn.fit(X_train, y_train)

    train_accuracies[neighbor] = knn.score(X_train, y_train)
    test_accuracies[neighbor] = knn.score(X_test, y_test)

print("Accuracy on train:", train_accuracies, '\n', "Accuracy on test:", test_accuracies)

plt.title("KNN: Accuracy vs. Number of Neighbors")

plt.plot(neighbors, list(train_accuracies.values()), label="Train Accuracy", marker='o')

plt.plot(neighbors, list(test_accuracies.values()), label="Test Accuracy", marker='o')

plt.legend()

plt.xlabel("Number of Neighbors")
plt.ylabel("Accuracy")

plt.show()
