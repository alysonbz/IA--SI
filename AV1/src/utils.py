import pandas as pd
import numpy as np
import math
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression , LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def cancer_dataset():
    df = pd.read_csv(r"C:\Users\aryel\OneDrive\Documentos\IA_ary\IA--SI\AV1\dataset\Cancer_Data.csv")
    return df