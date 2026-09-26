# train_optimized.py
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, LeakyReLU
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import joblib
import os

def load_dataset():
    data_path = Path(r"D:\deeplearning\project\CRM2\sales_data_sample.csv")
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at: {data_path}")
    df = pd.read_csv(data_path, encoding='cp1252')
    print("Dataset loaded successfully!")
    return df

def preprocess_data(df):
    target = 'SALES'
    features = ['QUANTITYORDERED', 'PRICEEACH', 'ORDERLINENUMBER', 
                'PRODUCTLINE', 'MSRP', 'PRODUCTCODE', 'COUNTRY', 'TERRITORY', 
                'MONTH_ID', 'YEAR_ID']

    X = df[features]
    y = df[target]

    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()

    numeric_transformer = Pipeline([
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline([
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

    X_processed = preprocessor.fit_transform(X)
    print("Preprocessing done!")

    os.makedirs("models", exist_ok=True)
    joblib.dump(preprocessor, "models/preprocessor.pkl")
    return X_processed, y

def build_model(input_dim):
    model = Sequential()
    model.add(Dense(256, input_dim=input_dim))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    model.add(Dense(128))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    model.add(Dense(64))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(0.1))

    model.add(Dense(1, activation='linear'))
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    print("Complex model built successfully!")
    return model

def main():
    print("Loading dataset...")
    df = load_dataset()

    print("Preprocessing data...")
    X, y = preprocess_data(df)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Building model...")
    model = build_model(X_train.shape[1])

    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    checkpoint = ModelCheckpoint("models/best_crm_model.h5", monitor='val_loss', save_best_only=True)

    print("Training model...")
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=100,
        batch_size=32,
        callbacks=[early_stop, checkpoint]
    )

    print("Training complete!")
    model.save("models/crm_sales_model_final.h5")
    print("Final model saved to models/crm_sales_model_final.h5")

    loss, mae = model.evaluate(X_test, y_test)
    print(f"Test MAE: {mae}")

if __name__ == "__main__":
    main()
