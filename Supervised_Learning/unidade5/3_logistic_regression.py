from src.utils import load_diabetes_clean_dataset
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression  # Correct import

diabetes_df = load_diabetes_clean_dataset()

# Features and target variable
X = diabetes_df.drop(['diabetes'], axis=1)
y = diabetes_df['diabetes'].values

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Instantiate the Logistic Regression model
logreg = LogisticRegression()

# Fit the model
logreg.fit(X_train, y_train)

# Predict probabilities for the test set
y_pred_probs = logreg.predict_proba(X_test)  # Provide X_test here

# Print the probabilities for the first 10 samples
print(y_pred_probs[:10])
