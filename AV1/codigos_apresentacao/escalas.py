from inconsistencias import X_limpo, y_limpo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, minmax_scale
import pandas as pd

scaler = StandardScaler()

variaveis_numericas = [
    "age",
    "trestbps",
    "chol",
    "thalach",
    "oldpeak"
]

variaveis_categoricas = [
    "sex",
    "cp",
    "fbs",
    "restecg",
    "exang",
    "slope",
    "ca",
    "thal"
]

X_treino, X_teste, y_treino, y_teste = train_test_split(X_limpo, y_limpo, test_size=0.2, random_state=42, stratify=y_limpo)

X_treino_numerico = scaler.fit_transform(X_treino[variaveis_numericas])

X_teste_numerico = scaler.transform(X_teste[variaveis_numericas])

encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")

X_treino_categorico = encoder.fit_transform(X_treino[variaveis_categoricas])

X_teste_categorico = encoder.transform(X_teste[variaveis_categoricas])

X_treino_numerico = pd.DataFrame(
    X_treino_numerico,
    columns=variaveis_numericas,
    index=X_treino.index
)

X_teste_numerico = pd.DataFrame(
    X_teste_numerico,
    columns=variaveis_numericas,
    index=X_teste.index
)

X_treino_categorico = pd.DataFrame(
    X_treino_categorico,
    columns=encoder.get_feature_names_out(variaveis_categoricas),
    index=X_treino.index
)

X_teste_categorico = pd.DataFrame(
    X_teste_categorico,
    columns=encoder.get_feature_names_out(variaveis_categoricas),
    index=X_teste.index
)

X_treino_final = pd.concat([X_treino_numerico, X_treino_categorico], axis=1)

X_teste_final = pd.concat([X_teste_numerico, X_teste_categorico], axis=1)

dataset_limpo_debug = False

if dataset_limpo_debug:
    # Verificando as dimensões
    print("\nDimensão dos dados de treino:")
    print(X_treino_final.shape)

    print("\nDimensão dos dados de teste:")
    print(X_teste_final.shape)

    print("\nDimensão de y_treino:")
    print(y_treino.shape)

    print("\nDimensão de y_teste:")
    print(y_teste.shape)

