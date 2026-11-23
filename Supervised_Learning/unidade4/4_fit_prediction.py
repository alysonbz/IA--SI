from src.utils import load_sales_clean_dataset
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
sales_df = load_sales_clean_dataset()

# Import mean_squared_error (Corrigido o nome da função)
from sklearn.metrics import mean_squared_error

# Create X and y arrays (Corrigido para excluir as colunas indicadas e extrair a label)
X = sales_df.drop(columns=["sales", "influencer"]).values
y = sales_df["sales"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Instantiate the model (Corrigido: adicionado parênteses para inicializar)
reg = LinearRegression()

# Fit the model to the data
reg.fit(X_train, y_train)

# Make predictions
y_pred = reg.predict(X_test)
print("Predictions: {}, Actual Values: {}".format(y_pred[:2], y_test[:2]))

# Compute R-squared (Corrigido: método correto é .score)
r_squared = reg.score(X_test, y_test)

# Compute RMSE (Corrigido o nome da função)
# Nota: Nas versões recentes do scikit-learn, você também pode calcular fazendo: root_mean_squared_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)

# Print the metrics
print("R^2: {}".format(r_squared))
print("RMSE: {}".format(rmse))