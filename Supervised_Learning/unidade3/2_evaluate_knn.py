from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from src.utils import load_churn_dataset

churn_df = load_churn_dataset()

X = churn_df[["account_length",  "total_day_charge" , "total_eve_charge",  "total_night_charge","total_intl_charge","number_customer_service_calls"]].values
y = churn_df["churn"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(X_train, y_train)

print(knn.score(X_test, y_test))
