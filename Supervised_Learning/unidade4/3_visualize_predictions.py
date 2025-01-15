
from src.utils import processing_sales_clean
# Import matplotlib.pyplot
<<<<<<< HEAD
import matplotlib.pyplot as plt
=======
import pyplot.matplotlib as plt
>>>>>>> 1239a00c96cd4d3adea696c64633d52b04d5adf1

X,y,predictions = processing_sales_clean()

# Create scatter plot
plt.scatter(X, y, color="blue")

# Create line plot
<<<<<<< HEAD
plt.plot(X, predictions, color="red")
=======
plt.plot(X, y, color="red")
>>>>>>> 1239a00c96cd4d3adea696c64633d52b04d5adf1
plt.xlabel("Radio Expenditure ($)")
plt.ylabel("Sales ($)")

# Display the plot
plt.show()