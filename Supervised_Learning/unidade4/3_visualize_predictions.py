
from src.utils import processing_sales_clean
# Import matplotlib.pyplot
import matplotlib.pyplot as plt

X,y,predictions = processing_sales_clean()

# Create scatter plot
plt.scatter(X_bmi, y, color="Blue")

# Create line plot
plt.plot(X_bmi, predictions, color="Red")
plt.xlabel("Radio Expenditure ($)")
plt.ylabel("Sales ($)")

# Display the plot
plt.show()