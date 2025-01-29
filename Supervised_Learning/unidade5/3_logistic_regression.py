from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from src.utils import load_diabetes_clean_dataset

# Load and prepare the dataset
diabetes_df = load_diabetes_clean_dataset()
X = diabetes_df.drop(['diabetes'], axis=1)
y = diabetes_df['diabetes'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Instantiate the model with a different solver and increased iterations
logreg = LogisticRegression(max_iter=500, solver='liblinear')

# Fit the model
logreg.fit(X_train_scaled, y_train)

# Predict probabilities
y_pred_probs = logreg.predict_proba(X_test_scaled)[:, 1]  # Probabilities for the positive class

print(y_pred_probs[:10])
