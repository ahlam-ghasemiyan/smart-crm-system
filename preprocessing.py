
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from pathlib import Path


class CRMPreprocessor:
    def __init__(self):
        self.transformer = None
        self.numeric_features = [
            "QUANTITYORDERED",
            "PRICEEACH",
            "ORDERLINENUMBER",
            "SALES",
            "MSRP",
            "MONTH_ID",
            "YEAR_ID"
        ]

        self.categorical_features = [
            "PRODUCTLINE",
            "PRODUCTCODE",
            "COUNTRY",
            "TERRITORY",
            "DEALSIZE",
            "STATUS"
        ]

    def load_data(self, filepath: str) -> pd.DataFrame:
        df = pd.read_csv(filepath)
        return df

    def build_pipeline(self):
        numeric_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])

        categorical_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])

        self.transformer = ColumnTransformer(
            transformers=[
                ("num", numeric_pipeline, self.numeric_features),
                ("cat", categorical_pipeline, self.categorical_features),
            ]
        )

    def fit_transform(self, df: pd.DataFrame):
        """برای آموزش مدل استفاده می‌شود"""
        self.build_pipeline()
        X = df[self.numeric_features + self.categorical_features]
        X_transformed = self.transformer.fit_transform(X)
        return X_transformed

    def transform(self, df: pd.DataFrame):
        """برای پیش‌بینی استفاده می‌شود"""
        X = df[self.numeric_features + self.categorical_features]
        X_transformed = self.transformer.transform(X)
        return X_transformed

    def save(self, path="models/preprocessor.pkl"):
        Path("models").mkdir(exist_ok=True)
        joblib.dump(self.transformer, path)

    def load(self, path="models/preprocessor.pkl"):
        self.transformer = joblib.load(path)
        return self.transformer


def split_data(df: pd.DataFrame, target: str):
    X = df.drop(columns=[target])
    y = df[target]

    return train_test_split(X, y, test_size=0.2, random_state=42)
