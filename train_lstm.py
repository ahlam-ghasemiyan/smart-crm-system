from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os

def load_dataset():
    data_path = Path(r"D:\project\CRM2\sales_data_sample.csv")
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at: {data_path}")

    df = pd.read_csv(data_path, encoding='cp1252')
    df['ORDERDATE'] = pd.to_datetime(df['ORDERDATE'])
    df = df.sort_values('ORDERDATE')
    print("Dataset loaded and sorted by date successfully!")
    return df

#  ایجاد sequences برای LSTM 
def create_sequences(data, seq_length=30):
    X, y = [], []
    for i in range(len(data)-seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

def preprocess_series(df, seq_length=30):
    sales_series = df['SALES'].values
    X, y = create_sequences(sales_series, seq_length)

    # reshape برای LSTM: [samples, timesteps, features]
    X = X.reshape((X.shape[0], X.shape[1], 1))
    y = y.reshape(-1,1)
    print(f"Created {X.shape[0]} sequences for LSTM with sequence length {seq_length}.")
    return X, y

# تعریف مدل LSTM 
def build_model(seq_length):
    model = Sequential()
    model.add(LSTM(64, input_shape=(seq_length,1), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(32))
    model.add(Dropout(0.2))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    print("LSTM model built successfully!")
    return model

def main():
    df = load_dataset()
    seq_length = 30
    X, y = preprocess_series(df, seq_length)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = build_model(seq_length)

    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    os.makedirs("models", exist_ok=True)
    checkpoint = ModelCheckpoint("models/best_lstm_sales_model.h5", monitor='val_loss', save_best_only=True)

    print("Training LSTM model...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=50,
        batch_size=16,
        callbacks=[early_stop, checkpoint]
    )

    model.save("models/lstm_sales_model_final.h5")
    print("Final LSTM model saved to models/lstm_sales_model_final.h5")

    loss, mae = model.evaluate(X_test, y_test)
    print(f"Test MAE: {mae}")

if __name__ == "__main__":
    main()
