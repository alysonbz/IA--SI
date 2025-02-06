from src.utils import log_reg_diabetes
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score  # Importa roc_auc_score

y_prob, y_test, y_pred = log_reg_diabetes()

# Calcula o roc_auc_score
print(roc_auc_score(y_test, y_prob))

# Calcula a matriz de confusão
print(confusion_matrix(y_test, y_pred))

# Calcula o relatório de classificação (precisão, recall, F1-score)
print(classification_report(y_test, y_pred))
