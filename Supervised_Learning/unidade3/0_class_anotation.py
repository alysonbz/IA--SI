from pandas.core.algorithms import value_counts_arraylike

from src.utils import load_churn_dataset
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

churn_df = load_churn_dataset()
print(churn_df['internation_plan'])

