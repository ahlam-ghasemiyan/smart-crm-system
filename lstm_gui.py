import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from pathlib import Path

model_path = Path("models/lstm_sales_model_final.h5")
if not model_path.exists():
    raise FileNotFoundError(f"LSTM model not found at: {model_path}")
model = load_model(model_path)

def load_dataset():
    file_path = filedialog.askopenfilename(filetypes=[("CSV files","*.csv")])
    if not file_path:
        return None
    df = pd.read_csv(file_path, encoding='cp1252')
    df['ORDERDATE'] = pd.to_datetime(df['ORDERDATE'])
    df = df.sort_values('ORDERDATE')
    return df

def create_sequences(data, seq_length=30):
    X, y = [], []
    for i in range(len(data)-seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

def predict_future(df, days=7, seq_length=30):
    sales_series = df['SALES'].values
    history = sales_series[-seq_length:]  # آخرین seq_length روز
    predictions = []

    for _ in range(days):
        X_input = np.array(history[-seq_length:]).reshape((1, seq_length, 1))
        pred = model.predict(X_input, verbose=0)[0][0]
        predictions.append(pred)
        history = np.append(history, pred)
    return predictions

def plot_predictions(real=None, pred=None):
    fig, ax = plt.subplots(figsize=(6,4))
    if real is not None:
        ax.plot(range(len(real)), real, label="Actual Sales")
    if pred is not None:
        ax.plot(range(len(pred)), pred, label="Predicted Sales", linestyle='--')
    ax.set_xlabel("Time")
    ax.set_ylabel("Sales")
    ax.legend()
    return fig

class SalesPredictorGUI:
    def __init__(self, master):
        self.master = master
        master.title("Sales Predictor (LSTM)")

        self.df = None

        self.load_button = tk.Button(master, text="Load CSV", command=self.load_csv)
        self.load_button.pack(pady=5)

        self.days_label = tk.Label(master, text="Predict next days:")
        self.days_label.pack()
        self.days_entry = tk.Entry(master)
        self.days_entry.insert(0,"7")
        self.days_entry.pack(pady=5)

        self.predict_button = tk.Button(master, text="Predict", command=self.predict)
        self.predict_button.pack(pady=5)

        self.canvas_frame = tk.Frame(master)
        self.canvas_frame.pack()

    def load_csv(self):
        df = load_dataset()
        if df is not None:
            self.df = df
            messagebox.showinfo("Info", "Dataset loaded successfully!")

    def predict(self):
        if self.df is None:
            messagebox.showwarning("Warning", "Please load a dataset first!")
            return
        try:
            days = int(self.days_entry.get())
        except:
            messagebox.showerror("Error", "Invalid number of days!")
            return

        preds = predict_future(self.df, days=days)
        fig = plot_predictions(pred=preds)
        
        # نمایش در Tkinter
        for widget in self.canvas_frame.winfo_children():
            widget.destroy()
        canvas = FigureCanvasTkAgg(fig, master=self.canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack()

if __name__ == "__main__":
    root = tk.Tk()
    app = SalesPredictorGUI(root)
    root.mainloop()
