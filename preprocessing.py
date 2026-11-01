import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

ZERO_AS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
FEATURES = [
    "Glucose",
    "BMI",
    "Age",
    "DiabetesPedigreeFunction",
    "Pregnancies"
]
TARGET = "Outcome"


def load_data(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.replace({pd.NA: np.nan})
    df = df.dropna(subset=[TARGET])
    df[TARGET] = df[TARGET].astype(int)

    # 1. Nilai 0 di kolom medis diganti median
    for col in ZERO_AS_MISSING:
        df[col] = df[col].replace(0, df[col].median())

    # 2. Outlier clipping pakai IQR
    for col in FEATURES:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        df[col] = df[col].clip(lower, upper)

    return df


def split_data(df):
    X = df[FEATURES]
    y = df["Outcome"]

    return train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=160,
        stratify=y
    )

def make_preprocessor():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])