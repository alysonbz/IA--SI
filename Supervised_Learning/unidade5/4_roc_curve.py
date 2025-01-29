import matplotlib.pyplot as plt
from src.utils import log_reg_diabetes
from sklearn.metrics import roc_curve  # Import roc_curve

y_prob, y_test, _ = log_reg_diabetes()

# Generate ROC curve values: fpr, tpr, thresholds
fpr, tpr, thresholds = roc_curve(y_test, y_prob)

# Plot the diagonal line for random predictions (no discrimination)
plt.plot([0, 1], [0, 1], 'k--')

# Plot tpr against fpr (ROC curve)
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Diabetes Prediction')
plt.show()
