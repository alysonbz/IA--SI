from inconsistencias import heart_limpo, X_limpo, y_limpo, variaveis_numericas

print("\nAnálise estatística das variáveis numéricas:")
print(X_limpo[variaveis_numericas].describe().T)

print("\nDistribuição das classes y:")
print(y_limpo.value_counts().sort_index())

print("\nPorcentagem das classes de y:")
print(y_limpo.value_counts(normalize=True).sort_index() * 100)