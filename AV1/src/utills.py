import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression , LogisticRegression
from sklearn.model_selection import train_test_split
def kc_house_dataset():
    df = pd.read_csv(r"C:\Users\Paulo\Desktop\IA-SI\IA--SI\AV1\dataset_kc_house\kc_house_data.csv")
    return df

def diabetes_dataset():
    df = pd.read_csv(r"C:\Users\Paulo\Desktop\IA-SI\IA--SI\AV1\dataset_diabetes\diabetes.csv")
    return df

def diabetes_ajustado_dataset():
    df = pd.read_csv(r"C:\Users\Paulo\Desktop\IA-SI\IA--SI\AV1\dataset_diabetes_ajustado\diabetes_ajustado.csv")
    return df

def houses_sales_processed_dataset():
    df = pd.read_csv(r"C:\Users\Paulo\Desktop\IA-SI\IA--SI\AV1\dataset_houses_sales_processed\houses_sales_processed.csv")
    return df