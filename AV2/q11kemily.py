# ==========================================
# QUESTÃO 11 - Análise Integrada dos Resultados
# ==========================================
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

from _dados_ import energy
from q5 import kmeans_final, dados_cluster
from _q8_treinamento import X_test_scaled_df, y_test
from q10 import pred_linear, pred_ridge, pred_lasso

print("="*60)
print("QUESTÃO 11 - ANÁLISE INTEGRADA")
print("="*60)

dados_cluster['Y1'] = energy['Y1']

print("\n--- 11.a e 11.b: MÉDIAS DAS VARIÁVEIS POR CLUSTER ---")
medias_clusters = dados_cluster.groupby('cluster')[['X1', 'X2', 'X5', 'Y1']].mean().round(2)
print(medias_clusters)

plt.figure(figsize=(8, 5))
sns.boxplot(data=dados_cluster, x='cluster', y='Y1', palette='Set2')
plt.title('Distribuição da Carga de Aquecimento (Y1) por Cluster', fontsize=14)
plt.xlabel('Cluster')
plt.ylabel('Y1 (kWh/m²)')
plt.show()

print("\n--- 11.d: ERRO DOS MODELOS POR CLUSTER ---")

clusters_teste = kmeans_final.predict(X_test_scaled_df)

df_teste = pd.DataFrame({
    'Real': y_test,
    'Cluster': clusters_teste,
    'Pred_Linear': pred_linear,
    'Pred_Ridge': pred_ridge,
    'Pred_Lasso': pred_lasso
})

def calcular_erros(df, pred_col, nome):
    print(f"\nModelo: {nome}")
    resultados = []
    for c in sorted(df['Cluster'].unique()):
        sub = df[df['Cluster'] == c]
        if len(sub) > 0:
            mae = mean_absolute_error(sub['Real'], sub[pred_col])
            rmse = np.sqrt(mean_squared_error(sub['Real'], sub[pred_col]))
            resultados.append({'Cluster': c, 'Amostras': len(sub), 'MAE': round(mae, 2), 'RMSE': round(rmse, 2)})
    print(pd.DataFrame(resultados))

calcular_erros(df_teste, 'Pred_Linear', 'Regressão Linear')
calcular_erros(df_teste, 'Pred_Ridge', 'Ridge')
calcular_erros(df_teste, 'Pred_Lasso', 'Lasso')

# ==========================================
# 11.e) Conclusões (Texto para o Slide)
# ==========================================
print("\n" + "="*60)
print("11.e) CONCLUSÕES DA ANÁLISE INTEGRADA")
print("="*60)
