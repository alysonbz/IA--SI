import pandas as pd
from sklearn.linear_model import LinearRegression
from _preparacao_ import *

reg = LinearRegression()

reg.fit(X_train_scaled_df, y_train)
predicao = reg.predict(X_test_scaled_df)

if __name__ == "__main__":
    print("\nIntercepto:")
    print(reg.intercept_)

    print("\nCoeficientes:")
    coeficientes = pd.DataFrame({
        'Variável': variaveis_preditoras,
        'Coeficiente': reg.coef_
    })

    print(coeficientes)

    print("\nPrimeiras previsões:")
    print(predicao[:10])
