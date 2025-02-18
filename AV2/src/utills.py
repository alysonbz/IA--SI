import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression , LogisticRegression
from sklearn.model_selection import train_test_split
def diabetes_dataset():
    df = pd.read_csv(r"C:\Users\nater\OneDrive\Desktop\IA-SI\IA--SI\AV2\dataset_diabetes\diabetes.csv")
    return df