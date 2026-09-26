# train_hyperopt.py
from pathlib import Path
import pandas as pd
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
import keras_tuner as kt

def load_dataset():
    data_path = Path(r"D:\project\CRM2\sales_data_sample.csv")
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

def build_model(hp):
    model = Sequential()
    for i in range(hp.Int('num_layers', 2, 5)):
        units = hp.Int(f'units_{i}', min_value=32, max_value=256, step=32)
        model.add(Dense(units))
        model.add(LeakyReLU(alpha=0.1))
        model.add(BatchNormalization())
        dropout_rate = hp.Float(f'dropout_{i}', 0.0, 0.5, step=0.1)
        model.add(Dropout(dropout_rate))

    model.add(Dense(1, activation='linear'))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            hp.Float('learning_rate', 1e-4, 1e-2, sampling='log')
        ),
        loss='mse',
        metrics=['mae']
    )
    return model

def main():
    df = load_dataset()
    X, y = preprocess_data(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    tuner = kt.RandomSearch(
        build_model,
        objective='val_mae',
        max_trials=20, 
        executions_per_trial=1,
        directory='models',
        project_name='crm_hyperopt'
    )

    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    tuner.search(X_train, y_train,
                 validation_split=0.2,
                 epochs=50,
                 batch_size=32,
                 callbacks=[early_stop])

    best_model = tuner.get_best_models(num_models=1)[0]
    best_model.save("models/best_hyperopt_crm_model.h5")
    print("Best hyperparameter-tuned model saved!")

    loss, mae = best_model.evaluate(X_test, y_test)
    print(f"Test MAE: {mae}")

if __name__ == "__main__":
    main()
