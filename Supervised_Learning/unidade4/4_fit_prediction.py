from src.utils import load_sales_clean_dataset
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

sales_df = load_sales_clean_dataset()

# Criação dos arrays X e Y
X = sales_df.drop(["sales","influencer"], axis=1)
y = sales_df["sales"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Instanciar o modelo de regressão linear
reg = LinearRegression()

# Ajustar o modelo aos dados
reg.fit(X_train,y_train)

# Previsões
y_pred = reg.predict(X_test)
print("Predictions: {}, Actual Values: {}".format(y_pred[:2], y_test[:2]))

# Calcular R-quadrado
r_squared = reg.score(X_test, y_test)

# Calcular RMSE
rmse = mean_squared_error(y_pred, y_test, squared=False)

# imprimir as métricas
print("R^2: {}".format(r_squared))
print("RMSE: {}".format(rmse))