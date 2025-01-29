from src.utils import log_reg_diabetes
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score  # Importa o roc_auc_score

y_prob, y_test, y_pred = log_reg_diabetes()

# Calcular o roc_auc_score
print(roc_auc_score(y_test, y_prob))  # AUC-ROC, passando os rótulos reais e as probabilidades previstas

# Calcular a matriz de confusão
print(confusion_matrix(y_test, y_pred))  # Matriz de confusão, passando os rótulos reais e as previsões

# Calcular o relatório de classificação
print(classification_report(y_test, y_pred))  # Relatório de classificação, com métricas como precisão, recall e F1-score
