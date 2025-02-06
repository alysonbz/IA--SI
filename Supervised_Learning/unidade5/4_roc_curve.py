import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from src.utils import log_reg_diabetes

# Importing the model and getting predictions
y_prob, y_test, _ = log_reg_diabetes()

# Generate ROC curve values: fpr, tpr, thresholds
fpr, tpr, thresholds = roc_curve(y_test, y_prob)

# Plot the ROC curve
plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line for random guess
plt.plot(fpr, tpr, label='ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Diabetes Prediction')
plt.legend()
plt.show()
