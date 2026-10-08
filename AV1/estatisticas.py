from inconsistencias import heart_limpo, X_limpo, y_limpo

print("\nAnálise estatística de X:")
print(X_limpo.describe().T)

print("\nDistribuição das classes y:")
print(y_limpo.value_counts().sort_index())

print("\nPorcentagem das classes de y:")
print(y_limpo.value_counts(normalize=True).sort_index() * 100)