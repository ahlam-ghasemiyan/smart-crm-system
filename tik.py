
# import tkinter as tk
# from tkinter import ttk, messagebox
# import pandas as pd
# import numpy as np
# from pathlib import Path
# import joblib
# from tensorflow.keras.models import load_model
# from tensorflow.keras.losses import MeanSquaredError
# import matplotlib.pyplot as plt
# from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
# import matplotlib.dates as mdates

# mlp_model_path = Path(r"D:\deeplearning\project\CRM2\models\crm_sales_model_final.h5")
# lstm_model_path = Path(r"D:\deeplearning\project\CRM2\models\lstm_sales_model_final.h5")
# preprocessor_path = Path(r"D:\deeplearning\project\CRM2\models\preprocessor.pkl")
# dataset_path = Path(r"D:\deeplearning\project\CRM2\sales_data_sample.csv")

# df = pd.read_csv(dataset_path, encoding='cp1252')
# df['ORDERDATE'] = pd.to_datetime(df['ORDERDATE'])
# if 'YEAR_ID' not in df.columns:
#     df['YEAR_ID'] = df['ORDERDATE'].dt.year
# if 'MONTH_ID' not in df.columns:
#     df['MONTH_ID'] = df['ORDERDATE'].dt.month

# mlp_model = None
# lstm_model = None
# preprocessor = None
# try:
#     mlp_model = load_model(mlp_model_path, custom_objects={'mse': MeanSquaredError()})
# except Exception:
#     mlp_model = None

# try:
#     lstm_model = load_model(lstm_model_path, compile=False)
#     lstm_model.compile(optimizer='adam', loss='mse', metrics=['mse'])
# except Exception:
#     lstm_model = None

# try:
#     preprocessor = joblib.load(preprocessor_path)
# except Exception:
#     preprocessor = None

# def preprocess_mlp(df_):
#     features = ['QUANTITYORDERED', 'PRICEEACH', 'ORDERLINENUMBER',
#                 'PRODUCTLINE', 'MSRP', 'PRODUCTCODE', 'COUNTRY', 'TERRITORY',
#                 'MONTH_ID', 'YEAR_ID']
#     X = df_[features]
#     X_processed = preprocessor.transform(X)
#     return X_processed

# def predict_mlp(df_):
#     if mlp_model is None or preprocessor is None:
#         raise RuntimeError("MLP model or preprocessor not loaded.")
#     X = preprocess_mlp(df_)
#     preds = mlp_model.predict(X, verbose=0)
#     return preds

# def predict_lstm_future(df_, days=30, seq_length=30):
#     if lstm_model is None:
#         raise RuntimeError("LSTM model not loaded.")
#     series = df_['SALES'].values
#     history = series[-seq_length:]
#     preds = []
#     for _ in range(days):
#         X_input = np.array(history[-seq_length:]).reshape((1, seq_length,1))
#         pred = lstm_model.predict(X_input, verbose=0)[0][0]
#         preds.append(pred)
#         history = np.append(history, pred)
#     return preds

# class FutureSalesFrame(tk.Frame):
#     def __init__(self, master, df, **kwargs):
#         super().__init__(master, **kwargs)
#         self.df = df
#         tk.Label(self, text="پیش‌بینی آینده فروش محصول", font=("Arial", 20, "bold")).pack(pady=30)
#         center_frame = tk.Frame(self)
#         center_frame.pack(pady=20)
#         tk.Label(center_frame, text="محصول:", font=("Arial", 14)).grid(row=0, column=0, padx=10, pady=10)
#         self.product_combo = ttk.Combobox(center_frame, values=list(df['PRODUCTLINE'].unique()), font=("Arial", 12), width=25)
#         self.product_combo.grid(row=0, column=1, padx=10, pady=10)
#         if len(self.product_combo['values'])>0:
#             self.product_combo.current(0)
#         tk.Label(center_frame, text="کشور:", font=("Arial", 14)).grid(row=1, column=0, padx=10, pady=10)
#         self.country_combo = ttk.Combobox(center_frame, values=list(df['COUNTRY'].unique()), font=("Arial", 12), width=25)
#         self.country_combo.grid(row=1, column=1, padx=10, pady=10)
#         if len(self.country_combo['values'])>0:
#             self.country_combo.current(0)
#         self.prob_label = tk.Label(self, text="به احتمال ... درصد در ۳۰ روز آینده فروش خواهد داشت", font=("Arial", 14), fg="blue")
#         self.prob_label.pack(pady=10)
#         self.predict_btn = tk.Button(self, text="پیش‌بینی", bg="#28a745", fg="white",
#                                      font=("Arial", 14, "bold"), relief="flat", command=self.run_prediction)
#         self.predict_btn.pack(pady=20, ipadx=30, ipady=10)
#         self.canvas_frame = tk.Frame(self)
#         self.canvas_frame.pack(pady=10, fill="both", expand=True)

#     def run_prediction(self):
#         product = self.product_combo.get()
#         country = self.country_combo.get()
#         df_filtered = self.df[(self.df['PRODUCTLINE']==product) & (self.df['COUNTRY']==country)]
#         if df_filtered.empty:
#             messagebox.showwarning("Warning", "هیچ داده‌ای برای محصول و کشور انتخاب شده یافت نشد.")
#             return

#         if lstm_model is None:
#             messagebox.showinfo("Info", "LSTM model not available — only plotting historical sales.")
#             fig, ax = plt.subplots(figsize=(8,5))
#             ax.plot(df_filtered['ORDERDATE'], df_filtered['SALES'].values, label="real sell")
#             ax.set_xlabel("Time")
#             ax.set_ylabel("Sales")
#             ax.legend()
#         else:
#             preds_lstm = predict_lstm_future(df_filtered, days=30)
#             last_sales = df_filtered['SALES'].values[-1]
#             avg_pred_30 = np.mean(preds_lstm)
#             prob_30 = ((avg_pred_30 - last_sales)/last_sales)*100
#             self.prob_label.config(text=f"به احتمال {prob_30:.2f}% در ۳۰ روز آینده فروش خواهد داشت")
#             fig, ax = plt.subplots(figsize=(8,5))
#             ax.plot(df_filtered['ORDERDATE'], df_filtered['SALES'].values, label="real sell")
#             future_x = pd.date_range(start=df_filtered['ORDERDATE'].max() + pd.Timedelta(days=1), periods=30, freq='D')
#             ax.plot(future_x, preds_lstm, '--', label="Next 30 days(LSTM)")
#             ax.set_xlabel("Time")
#             ax.set_ylabel("Sales")
#             ax.legend()

#         for widget in self.canvas_frame.winfo_children():
#             widget.destroy()
#         canvas = FigureCanvasTkAgg(fig, master=self.canvas_frame)
#         canvas.draw()
#         canvas.get_tk_widget().pack(fill="both", expand=True)


# class TopSalesFrame(tk.Frame):
#     def __init__(self, master, df, **kwargs):
#         super().__init__(master, **kwargs)
#         self.df = df
#         tk.Label(self, text="بیشترین فروش محصولات", font=("Arial", 20, "bold")).pack(pady=30)
#         center_frame = tk.Frame(self)
#         center_frame.pack(pady=20)
#         tk.Label(center_frame, text="محصول:", font=("Arial", 14)).grid(row=0, column=0, padx=10, pady=10)
#         self.product_combo = ttk.Combobox(center_frame, values=list(df['PRODUCTLINE'].unique()), font=("Arial", 12), width=25)
#         self.product_combo.grid(row=0, column=1, padx=10, pady=10)
#         if len(self.product_combo['values'])>0:
#             self.product_combo.current(0)
#         self.show_btn = tk.Button(self, text="نمایش", bg="#28a745", fg="white", font=("Arial", 14, "bold"),
#                                   relief="flat", command=self.show_top_sales)
#         self.show_btn.pack(pady=20, ipadx=30, ipady=10)
#         self.result_frame = tk.Frame(self)
#         self.result_frame.pack(pady=10, fill="x")
#         self.line1 = tk.Label(self.result_frame, text="", font=("Arial", 14, "bold"), fg="black", justify="center")
#         self.line1.pack()
#         self.line2 = tk.Label(self.result_frame, text="", font=("Arial", 14, "bold"), fg="green", justify="center")
#         self.line2.pack()
#         self.line3 = tk.Label(self.result_frame, text="", font=("Arial", 14, "bold"), fg="black", justify="center")
#         self.line3.pack()
#         self.canvas_frame = tk.Frame(self)
#         self.canvas_frame.pack(pady=10, fill="both", expand=True)

#     def show_top_sales(self):
#         product = self.product_combo.get()
#         df_filtered = self.df[self.df['PRODUCTLINE'] == product]
#         if df_filtered.empty:
#             messagebox.showwarning("Warning", "هیچ داده‌ای برای محصول انتخاب شده یافت نشد.")
#             return
#         country_sales = df_filtered.groupby('COUNTRY')['SALES'].sum().sort_values(ascending=False)
#         top_country = country_sales.idxmax()
#         top_sales = country_sales.max()
#         start_year = df_filtered['YEAR_ID'].min()
#         end_year = df_filtered['YEAR_ID'].max()
#         self.line1.config(text=f"{product} بیشترین فروش را در {top_country} داشته است.")
#         self.line2.config(text=f"مقدار فروش: {top_sales:,}")
#         self.line3.config(text=f"بازه سال‌ها: {start_year} تا {end_year}")
#         fig, ax = plt.subplots(figsize=(12,6))

#         country_sales.plot(kind='bar', ax=ax, color="#28a745")
#         ax.set_title(f"Sales of {product} by Country")
#         ax.set_xlabel("Country")
#         ax.set_ylabel("Sales")
#         plt.xticks(rotation=45)
#         for widget in self.canvas_frame.winfo_children():
#             widget.destroy()
#         canvas = FigureCanvasTkAgg(fig, master=self.canvas_frame)
#         canvas.draw()
#         canvas.get_tk_widget().pack(fill="both", expand=True)


# class CustomerOrdersFrame(tk.Frame):
#     def __init__(self, master, df, **kwargs):
#         super().__init__(master, **kwargs)
#         self.df = df
#         tk.Label(self, text="سفارشات مشتریان", font=("Arial", 20, "bold")).pack(pady=20)
#         top_frame = tk.Frame(self)
#         top_frame.pack(pady=5, fill="x")
#         tk.Label(top_frame, text="انتخاب مشتری:", font=("Arial", 14)).pack(side="left", padx=10)
#         customers = sorted(list(df["CUSTOMERNAME"].dropna().unique()))
#         self.customer_box = ttk.Combobox(top_frame, values=customers, font=("Arial", 12), width=40, state="readonly")
#         self.customer_box.pack(side="left", padx=10)
#         if customers:
#             self.customer_box.current(0)
#         load_btn = tk.Button(top_frame, text="Load Orders", bg="#3498db", fg="white", font=("Arial", 11, "bold"),
#                              relief="flat", command=self.load_customer_orders)
#         load_btn.pack(side="left", padx=10)
#         columns = ("order", "date", "sales", "country", "product", "status")
#         self.tree = ttk.Treeview(self, columns=columns, show="headings", height=12)
#         self.tree.pack(padx=10, pady=15, fill="both", expand=False)
#         self.tree.heading("order", text="Order No")
#         self.tree.heading("date", text="Order Date")
#         self.tree.heading("sales", text="Sales")
#         self.tree.heading("country", text="Country")
#         self.tree.heading("product", text="Product")
#         self.tree.heading("status", text="Status")
#         self.tree.column("order", width=100, anchor="center")
#         self.tree.column("date", width=130, anchor="center")
#         self.tree.column("sales", width=120, anchor="e")
#         self.tree.column("country", width=120, anchor="center")
#         self.tree.column("product", width=220, anchor="w")
#         self.tree.column("status", width=120, anchor="center")
#         self.info_label = tk.Label(self, text="", font=("Arial", 13), fg="black", justify="center")
#         self.info_label.pack(pady=10)

#     def load_customer_orders(self):
#         customer = self.customer_box.get()
#         data = self.df[self.df["CUSTOMERNAME"] == customer]
#         for r in self.tree.get_children():
#             self.tree.delete(r)
#         if data.empty:
#             messagebox.showinfo("Info", "هیچ سفارشی برای این مشتری وجود ندارد.")
#             self.info_label.config(text="")
#             return
#         data_sorted = data.sort_values("ORDERDATE", ascending=False)
#         for _, r in data_sorted.iterrows():
#             order_no = r.get("ORDERNUMBER", "")
#             date = r.get("ORDERDATE", "")
#             sales = r.get("SALES", 0)
#             country = r.get("COUNTRY", "")
#             product = r.get("PRODUCTLINE", "")
#             status = r.get("STATUS", "")
#             date_str = date.strftime("%Y-%m-%d") if not pd.isna(date) else ""
#             self.tree.insert("", "end", values=(order_no, date_str, f"{sales:,.0f}", country, product, status))
#         total_orders = len(data)
#         total_sales = data["SALES"].sum()
#         years = pd.to_datetime(data["ORDERDATE"]).dt.year
#         start_year = int(years.min()) if not years.isna().all() else ""
#         end_year = int(years.max()) if not years.isna().all() else ""
#         info_text = (
#             f'مشتری "{customer}" تا کنون {total_orders} سفارش ثبت کرده است.\n'
#             f'مجموع فروش: {total_sales:,.0f}$\n'
#             f'بازه فعالیت: {start_year} تا {end_year}'
#         )
#         self.info_label.config(text=info_text)


# class CustomersFrame(tk.Frame):
#     """بخش مشتریان: KPI، Top10، country distribution، growth"""
#     def __init__(self, master, df, **kwargs):
#         super().__init__(master, **kwargs)
#         self.df = df.copy()
#         tk.Label(self, text="تحلیل کلی مشتریان", font=("Arial", 22, "bold")).pack(pady=12)

#         kpi_frame = tk.Frame(self)
#         kpi_frame.pack(pady=6, fill="x", padx=12)

#         total_customers = self.df['CUSTOMERNAME'].nunique()
#         self.k1 = tk.Label(kpi_frame, text=f"Total Customers\n{total_customers}", font=("Arial", 12, "bold"), bd=1, relief="solid", padx=12, pady=8)
#         self.k1.pack(side="left", padx=8, ipadx=6, fill="y")

#         sales_by_customer = self.df.groupby('CUSTOMERNAME')['SALES'].sum()
#         avg_sales = sales_by_customer.mean() if not sales_by_customer.empty else 0
#         self.k2 = tk.Label(kpi_frame, text=f"Avg Sales / Customer\n{avg_sales:,.0f}$", font=("Arial", 12, "bold"), bd=1, relief="solid", padx=12, pady=8)
#         self.k2.pack(side="left", padx=8, ipadx=6, fill="y")

#         #  loyal customers (>=3 orders)
#         orders_count = self.df.groupby('CUSTOMERNAME')['ORDERNUMBER'].nunique()
#         loyal_count = (orders_count >= 3).sum()
#         self.k3 = tk.Label(kpi_frame, text=f"Loyal Customers\n{loyal_count}", font=("Arial", 12, "bold"), bd=1, relief="solid", padx=12, pady=8)
#         self.k3.pack(side="left", padx=8, ipadx=6, fill="y")

#         charts_frame = tk.Frame(self)
#         charts_frame.pack(fill="both", expand=True, padx=10, pady=8)

#         left = tk.Frame(charts_frame)
#         left.pack(side="left", fill="both", expand=True, padx=6)

#         tk.Label(left, text="Top 10 Customers (by Sales)", font=("Arial", 12, "bold")).pack(pady=6)
#         self.top10_canvas_frame = tk.Frame(left)
#         self.top10_canvas_frame.pack(fill="both", expand=True)

#         self.top_tree = ttk.Treeview(left, columns=("customer", "sales"), show="headings", height=6)
#         self.top_tree.heading("customer", text="Customer")
#         self.top_tree.heading("sales", text="Sales")
#         self.top_tree.column("customer", width=200)
#         self.top_tree.column("sales", width=120, anchor="e")
#         self.top_tree.pack(pady=6, fill="x")

#         right = tk.Frame(charts_frame)
#         right.pack(side="left", fill="both", expand=True, padx=6)

#         tk.Label(right, text="Sales by Country", font=("Arial", 12, "bold")).pack(pady=6)
#         self.country_canvas_frame = tk.Frame(right)
#         self.country_canvas_frame.pack(fill="both", expand=True)

#         tk.Label(right, text="Customers growth (monthly)", font=("Arial", 12, "bold")).pack(pady=6)
#         self.growth_canvas_frame = tk.Frame(right)
#         self.growth_canvas_frame.pack(fill="both", expand=True)

#         self.render_top10()
#         self.render_country_dist()
#         self.render_growth()

#     def render_top10(self):
#         sales_by_customer = self.df.groupby('CUSTOMERNAME')['SALES'].sum().sort_values(ascending=False)
#         top10 = sales_by_customer.head(10)
#         for r in self.top_tree.get_children():
#             self.top_tree.delete(r)
#         for cust, val in top10.items():
#             self.top_tree.insert("", "end", values=(cust, f"{val:,.0f}$"))

#         # plot
#         fig, ax = plt.subplots(figsize=(6,3.5))
#         top10[::-1].plot(kind='barh', ax=ax, color='#2c7fb8')
#         ax.set_xlabel("Sales")
#         ax.set_ylabel("")
#         ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
#         fig.tight_layout()
#         for widget in self.top10_canvas_frame.winfo_children():
#             widget.destroy()
#         canvas = FigureCanvasTkAgg(fig, master=self.top10_canvas_frame)
#         canvas.draw()
#         canvas.get_tk_widget().pack(fill="both", expand=True)

#     def render_country_dist(self):
#         sales_by_country = self.df.groupby('COUNTRY')['SALES'].sum().sort_values(ascending=False)
#         fig, ax = plt.subplots(figsize=(6,3.5))
#         sales_by_country.plot(kind='bar', ax=ax, color='#91cf60')
#         ax.set_xlabel("Country")
#         ax.set_ylabel("Sales")
#         ax.xaxis.set_tick_params(rotation=45)
#         ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
#         fig.tight_layout()
#         for widget in self.country_canvas_frame.winfo_children():
#             widget.destroy()
#         canvas = FigureCanvasTkAgg(fig, master=self.country_canvas_frame)
#         canvas.draw()
#         canvas.get_tk_widget().pack(fill="both", expand=True)

#     def render_growth(self):
#         tmp = self.df.copy()
#         tmp['year_month'] = tmp['ORDERDATE'].dt.to_period('M').dt.to_timestamp()
#         customers_month = tmp.groupby('year_month')['CUSTOMERNAME'].nunique()
#         fig, ax = plt.subplots(figsize=(6,3.5))
#         ax.plot(customers_month.index, customers_month.values, marker='o')
#         ax.set_xlabel("Month")
#         ax.set_ylabel("Unique Customers")
#         ax.xaxis.set_major_locator(mdates.AutoDateLocator())
#         ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
#         fig.autofmt_xdate(rotation=45)
#         fig.tight_layout()
#         for widget in self.growth_canvas_frame.winfo_children():
#             widget.destroy()
#         canvas = FigureCanvasTkAgg(fig, master=self.growth_canvas_frame)
#         canvas.draw()
#         canvas.get_tk_widget().pack(fill="both", expand=True)





# class CustomerInfoFrame(tk.Frame):
#     """نمایش اطلاعات مشتریان: کل مشتریان و وفاداران"""
#     def __init__(self, master, df, **kwargs):
#         super().__init__(master, **kwargs)
#         self.df = df.copy()

#         tk.Label(self, text="اطلاعات مشتریان", font=("Arial", 22, "bold")).pack(pady=12)

#         # فریم بالا: انتخاب نوع مشتری
#         top_frame = tk.Frame(self)
#         top_frame.pack(pady=10, fill="x")

#         tk.Label(top_frame, text="انتخاب نوع مشتری:", font=("Arial", 14)).pack(side="left", padx=10)

#         self.type_combo = ttk.Combobox(
#             top_frame,
#             values=["کل مشتریان", "وفاداران"],
#             font=("Arial", 12),
#             width=20,
#             state="readonly"
#         )
#         self.type_combo.pack(side="left", padx=10)
#         self.type_combo.current(0)

#         load_btn = tk.Button(
#             top_frame,
#             text="نمایش",
#             bg="#3498db",
#             fg="white",
#             font=("Arial", 11, "bold"),
#             relief="flat",
#             command=self.load_customers
#         )
#         load_btn.pack(side="left", padx=10)


#         columns = ("name", "phone", "country")
#         self.tree = ttk.Treeview(self, columns=columns, show="headings", height=20)
#         self.tree.pack(padx=10, pady=15, fill="both", expand=True)
#         self.tree.heading("name", text="نام مشتری")
#         self.tree.heading("phone", text="شماره تماس")
#         self.tree.heading("country", text="کشور")
#         self.tree.column("name", width=250)
#         self.tree.column("phone", width=150, anchor="center")
#         self.tree.column("country", width=120, anchor="center")

#     def load_customers(self):
#         selection = self.type_combo.get()

#         if selection == "کل مشتریان":
#             data = self.df[["CUSTOMERNAME", "PHONE", "COUNTRY"]].drop_duplicates()
#         elif selection == "وفاداران":
#             orders_count = self.df.groupby("CUSTOMERNAME")["ORDERNUMBER"].nunique()
#             loyal_customers = orders_count[orders_count >= 3].index
#             data = self.df[self.df["CUSTOMERNAME"].isin(loyal_customers)][["CUSTOMERNAME", "PHONE", "COUNTRY"]].drop_duplicates()
#         else:
#             data = pd.DataFrame(columns=["CUSTOMERNAME", "PHONE", "COUNTRY"])

#         for r in self.tree.get_children():
#             self.tree.delete(r)

#         for _, row in data.iterrows():
#             self.tree.insert("", "end", values=(row["CUSTOMERNAME"], row["PHONE"], row["COUNTRY"]))




# class ProductCustomerFrame(tk.Frame):
#     """تحلیل اینکه هر محصول را کدام مشتری احتمالاً دوباره می‌خرد،
#        و اینکه تولید مجدد آن محصول چقدر ارزش دارد."""
#     def __init__(self, master, df, **kwargs):
#         super().__init__(master, **kwargs)
#         self.df = df.copy()

#         tk.Label(self, text="هر محصول برای کدام مشتری؟", font=("Arial", 22, "bold")).pack(pady=12)

#         top = tk.Frame(self)
#         top.pack(pady=5)

#         tk.Label(top, text="انتخاب محصول:", font=("Arial", 14)).pack(side="left", padx=10)

#         products = sorted(self.df["PRODUCTCODE"].unique())
#         self.cmb = ttk.Combobox(top, values=products, width=25, state="readonly", font=("Arial", 12))
#         self.cmb.pack(side="left")
#         self.cmb.bind("<<ComboboxSelected>>", self.analyze)

#         columns = ("customer", "count", "sales", "score", "percent")
#         self.tree = ttk.Treeview(self, columns=columns, show="headings", height=18)
#         self.tree.pack(fill="both", expand=True, pady=15)

#         self.tree.heading("customer", text="مشتری")
#         self.tree.heading("count", text="تعداد خرید")
#         self.tree.heading("sales", text="کل فروش")
#         self.tree.heading("score", text="امتیاز خرید")
#         self.tree.heading("percent", text="٪ احتمال خرید")

#         self.result_lbl = tk.Label(self, text="", font=("Arial", 16, "bold"), fg="green")
#         self.result_lbl.pack(pady=10)

#     def analyze(self, event=None):
#         product = self.cmb.get()
#         df_p = self.df[self.df["PRODUCTCODE"] == product]

#         customer_counts = df_p.groupby("CUSTOMERNAME")["ORDERNUMBER"].nunique()
#         customer_sales = df_p.groupby("CUSTOMERNAME")["SALES"].sum()

#         similar = self.df[self.df["PRODUCTLINE"] == df_p["PRODUCTLINE"].iloc[0]]
#         similar_counts = similar.groupby("CUSTOMERNAME")["ORDERNUMBER"].nunique()

#         max_count = customer_counts.max()
#         max_sales = customer_sales.max()
#         max_similar = similar_counts.max() if len(similar_counts) > 0 else 1

#         results = []
#         for c in customer_counts.index:
#             score = (
#                 (customer_counts[c] / max_count) * 0.5 +
#                 (customer_sales[c] / max_sales) * 0.3 +
#                 (similar_counts.get(c, 0) / max_similar) * 0.2
#             )
#             results.append([c, customer_counts[c], customer_sales[c], round(score, 3), int(score * 100)])

#         for i in self.tree.get_children():
#             self.tree.delete(i)

#         avg_score = 0
#         if len(results) > 0:
#             for r in results:
#                 self.tree.insert("", "end", values=r)
#             avg_score = sum([r[3] for r in results]) / len(results)

#         expected_value = avg_score * df_p["QUANTITYORDERED"].mean() * df_p["PRICEEACH"].mean()
#         production_cost = df_p["PRICEEACH"].mean() * 0.6

#         worth = (expected_value / production_cost) * 100 if production_cost > 0 else 0

#         self.result_lbl.config(text=f"ارزش تولید مجدد این محصول ≈ {worth:.1f}%")






# class AnalyticsFrame(tk.Frame):
#     def __init__(self, parent, df):
#         super().__init__(parent)
        
#         self.df = df.copy()
        
#         canvas = tk.Canvas(self)
#         scrollbar = tk.Scrollbar(self, orient="vertical", command=canvas.yview)
#         scroll_frame = tk.Frame(canvas)

#         scroll_frame.bind(
#             "<Configure>",
#             lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
#         )

#         canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
#         canvas.configure(yscrollcommand=scrollbar.set)

#         canvas.pack(side="left", fill="both", expand=True)
#         scrollbar.pack(side="right", fill="y")

#         self.images = []

#         self.build_charts(scroll_frame)
    

#     def build_charts(self, container):

#         import matplotlib.pyplot as plt
#         import seaborn as sns
#         from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        
#         sns.set(style="whitegrid")

#         def add_chart(title, fig):
#             lbl = tk.Label(container, text=title, font=("Arial", 16, "bold"))
#             lbl.pack(pady=10)

#             canvas = FigureCanvasTkAgg(fig, master=container)
#             canvas.get_tk_widget().pack(pady=10)

#         # 1) Top 10 Products by Sales
#         fig1 = plt.figure(figsize=(8, 4))
#         top_products = (
#             self.df.groupby("PRODUCTLINE")["SALES"]
#             .sum()
#             .sort_values(ascending=False)
#             .head(10)
#         )
#         sns.barplot(x=top_products.values, y=top_products.index)
#         plt.title("Top 10 Products by Sales")
#         add_chart("پرفروش‌ترین محصولات", fig1)

#         fig2 = plt.figure(figsize=(8, 4))
#         trend = (
#             self.df.groupby("MONTH_ID")["SALES"]
#             .sum()
#             .sort_index()
#         )
#         sns.lineplot(x=trend.index, y=trend.values, marker="o")
#         plt.title("Monthly Sales Trend")
#         add_chart("روند فروش ماهیانه", fig2)

#         fig3 = plt.figure(figsize=(6, 6))
#         share = self.df.groupby("PRODUCTLINE")["SALES"].sum()
#         plt.pie(share, labels=share.index, autopct="%1.1f%%")
#         plt.title("Market Share")
#         add_chart("سهم هر محصول از بازار", fig3)

#         fig4 = plt.figure(figsize=(8, 4))
#         freq = self.df["CUSTOMERNAME"].value_counts()
#         sns.histplot(freq, bins=20)
#         plt.title("Customer Purchase Frequency")
#         add_chart("تکرار خرید مشتریان", fig4)

#         fig5 = plt.figure(figsize=(8, 4))
#         country_sales = (
#             self.df.groupby("COUNTRY")["SALES"]
#             .sum()
#             .sort_values(ascending=False)
#             .head(15)
#         )
#         sns.barplot(x=country_sales.values, y=country_sales.index)
#         plt.title("Sales by Country")
#         add_chart("فروش بر اساس کشور", fig5)


#         fig6 = plt.figure(figsize=(8, 6))
#         num_df = self.df.select_dtypes(include=['int', 'float'])
#         sns.heatmap(num_df.corr(), annot=True, cmap="coolwarm")
#         plt.title("Correlation Heatmap")
#         add_chart("هیت‌مپ همبستگی", fig6)


#         fig7 = plt.figure(figsize=(8, 4))
#         profit = (
#             self.df.groupby("PRODUCTLINE")["SALES"]
#             .mean()
#             .sort_values(ascending=False)
#         )
#         sns.barplot(x=profit.values, y=profit.index)
#         plt.title("Average Profit per Product")
#         add_chart("میانگین سود هر محصول", fig7)

# class CRMApp:
#     def __init__(self, master):
#         self.master = master
#         master.title("CRM برای شما")
#         master.geometry("1200x780")

#         # Sidebar
#         self.sidebar = tk.Frame(master, width=260, bg="#2c3e50")
#         self.sidebar.pack(side="left", fill="y")
#         tk.Label(self.sidebar, text="ابزارها:", bg="#2c3e50", fg="white", font=("Arial", 14, "bold")).pack(pady=20)

#         menu_items = [
#             "HOME",
#             "پیش‌بینی آینده فروش محصول",
#             "بیشترین فروش محصول",
#             "سفارشات مشتریان",
#             "تحلیل کلی مشتریان",
#             "اطلاعات مشتریان",
#             "هر محصول برای کدام مشتری؟",
#             "نمودارهای تحلیلی"
#         ]

#         for item in menu_items:
#             btn = tk.Button(self.sidebar, text=item, font=("Arial", 12), bg="#34495e", fg="white",
#                             relief="flat", command=lambda n=item: self.show_content(n))
#             btn.pack(fill="x", pady=5, padx=12)


#         self.main_frame = tk.Frame(master, bg="#ecf0f1")
#         self.main_frame.pack(side="left", fill="both", expand=True)

#         self.current_widget = None
#         self.show_home()

#     def clear_main(self):
#         for w in self.main_frame.winfo_children():
#             w.destroy()

#     def show_home(self):
#         self.clear_main()
#         tk.Label(self.main_frame, text="CRM برای شما", font=("Arial", 28, "bold")).pack(pady=40)
#         tk.Label(self.main_frame, text="از منوی سمت چپ یک بخش را انتخاب کنید.", font=("Arial", 14)).pack(pady=10)

#     def show_content(self, name):
#         self.clear_main()
#         if name == "HOME":
#             self.show_home()
#             return
#         elif name == "پیش‌بینی آینده فروش محصول":
#             widget = FutureSalesFrame(self.main_frame, df)
#         elif name == "بیشترین فروش محصول":
#             widget = TopSalesFrame(self.main_frame, df)
#         elif name == "سفارشات مشتریان":
#             widget = CustomerOrdersFrame(self.main_frame, df)
#         elif name == "تحلیل کلی مشتریان":
#             widget = CustomersFrame(self.main_frame, df)
#         elif name == "اطلاعات مشتریان":  # <-- اضافه شد
#             widget = CustomerInfoFrame(self.main_frame, df)
#         elif name == "هر محصول برای کدام مشتری؟":
#             widget = ProductCustomerFrame(self.main_frame, df)
#         elif name == "نمودارهای تحلیلی":
#             widget = AnalyticsFrame(self.main_frame, df)
    

#         else:
#             widget = tk.Label(self.main_frame, text=f"این قسمت مربوط به {name} است.", font=("Arial", 16))
#         widget.pack(fill="both", expand=True)


# if __name__ == "__main__":
#     root = tk.Tk()
#     app = CRMApp(root)
#     root.mainloop()









































import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
import os
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# کتابخانه‌های یادگیری عمیق (در صورت عدم نصب، برنامه کرش نکند)
try:
    from tensorflow.keras.models import load_model
    from tensorflow.keras.losses import MeanSquaredError
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("هشدار: کتابخانه tensorflow نصب نیست. بخش‌های پیش‌بینی کار نخواهند کرد.")

# ==========================================
# 1. تنظیمات مسیر و بارگذاری فایل‌ها (هوشمند)
# ==========================================

# مسیر پوشه جاری پروژه
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# فرض بر این است که پوشه models داخل پوشه جاری است
MODELS_DIR = os.path.join(BASE_DIR, 'models') 

# تعریف مسیر فایل‌ها
DATASET_PATH = os.path.join(BASE_DIR, "sales_data_sample.csv")
MLP_PATH = os.path.join(MODELS_DIR, "crm_sales_model_final.h5")
LSTM_PATH = os.path.join(MODELS_DIR, "lstm_sales_model_final.h5")
PREPROCESSOR_PATH = os.path.join(MODELS_DIR, "preprocessor.pkl")

print(f"📂 مسیر پروژه: {BASE_DIR}")

# --- بارگذاری دیتاست ---
df = pd.DataFrame()
try:
    if os.path.exists(DATASET_PATH):
        df = pd.read_csv(DATASET_PATH, encoding='cp1252')
        df['ORDERDATE'] = pd.to_datetime(df['ORDERDATE'])
        if 'YEAR_ID' not in df.columns:
            df['YEAR_ID'] = df['ORDERDATE'].dt.year
        if 'MONTH_ID' not in df.columns:
            df['MONTH_ID'] = df['ORDERDATE'].dt.month
        print("✅ دیتاست با موفقیت بارگذاری شد.")
    else:
        messagebox.showerror("خطا", f"فایل دیتاست پیدا نشد:\n{DATASET_PATH}")
except Exception as e:
    messagebox.showerror("خطا", f"مشکل در خواندن فایل CSV:\n{e}")

# --- بارگذاری مدل‌ها ---
mlp_model = None
lstm_model = None
preprocessor = None

if TF_AVAILABLE:
    try:
        if os.path.exists(MLP_PATH):
            mlp_model = load_model(MLP_PATH, custom_objects={'mse': MeanSquaredError()})
            print("✅ مدل MLP بارگذاری شد.")
    except Exception as e:
        print(f"❌ خطا در بارگذاری MLP: {e}")

    try:
        if os.path.exists(LSTM_PATH):
            lstm_model = load_model(LSTM_PATH, compile=False)
            lstm_model.compile(optimizer='adam', loss='mse', metrics=['mse'])
            print("✅ مدل LSTM بارگذاری شد.")
    except Exception as e:
        print(f"❌ خطا در بارگذاری LSTM: {e}")

try:
    if os.path.exists(PREPROCESSOR_PATH):
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        print("✅ پیش‌پردازشگر بارگذاری شد.")
except Exception as e:
    print(f"❌ خطا در بارگذاری Preprocessor: {e}")


# ==========================================
# 2. توابع کمکی پیش‌بینی
# ==========================================

def preprocess_mlp(df_):
    features = ['QUANTITYORDERED', 'PRICEEACH', 'ORDERLINENUMBER',
                'PRODUCTLINE', 'MSRP', 'PRODUCTCODE', 'COUNTRY', 'TERRITORY',
                'MONTH_ID', 'YEAR_ID']
    # بررسی وجود ستون‌ها
    missing = [col for col in features if col not in df_.columns]
    if missing:
        raise ValueError(f"ستون‌های زیر در داده موجود نیستند: {missing}")
        
    X = df_[features]
    if preprocessor:
        X_processed = preprocessor.transform(X)
        return X_processed
    return X

def predict_mlp(df_):
    if mlp_model is None or preprocessor is None:
        return None
    try:
        X = preprocess_mlp(df_)
        preds = mlp_model.predict(X, verbose=0)
        return preds
    except Exception as e:
        print(f"Error inside predict_mlp: {e}")
        return None

def predict_lstm_future(df_, days=30, seq_length=30):
    if lstm_model is None:
        return []
    
    series = df_['SALES'].values
    if len(series) < seq_length:
        # داده کافی برای پیش‌بینی وجود ندارد
        return []

    history = series[-seq_length:]
    preds = []
    try:
        for _ in range(days):
            X_input = np.array(history[-seq_length:]).reshape((1, seq_length, 1))
            pred = lstm_model.predict(X_input, verbose=0)[0][0]
            preds.append(pred)
            history = np.append(history, pred)
        return preds
    except Exception as e:
        print(f"Error inside predict_lstm: {e}")
        return []

# ==========================================
# 3. فریم‌های برنامه (GUI Frames)
# ==========================================

class FutureSalesFrame(tk.Frame):
    def __init__(self, master, df, **kwargs):
        super().__init__(master, **kwargs)
        self.df = df
        tk.Label(self, text="پیش‌بینی آینده فروش محصول", font=("Arial", 20, "bold")).pack(pady=30)
        
        center_frame = tk.Frame(self)
        center_frame.pack(pady=20)
        
        tk.Label(center_frame, text="محصول:", font=("Arial", 14)).grid(row=0, column=0, padx=10, pady=10)
        self.product_combo = ttk.Combobox(center_frame, values=list(df['PRODUCTLINE'].unique()) if not df.empty else [], font=("Arial", 12), width=25)
        self.product_combo.grid(row=0, column=1, padx=10, pady=10)
        if self.product_combo['values']:
            self.product_combo.current(0)
            
        tk.Label(center_frame, text="کشور:", font=("Arial", 14)).grid(row=1, column=0, padx=10, pady=10)
        self.country_combo = ttk.Combobox(center_frame, values=list(df['COUNTRY'].unique()) if not df.empty else [], font=("Arial", 12), width=25)
        self.country_combo.grid(row=1, column=1, padx=10, pady=10)
        if self.country_combo['values']:
            self.country_combo.current(0)
            
        self.prob_label = tk.Label(self, text="...", font=("Arial", 14), fg="blue")
        self.prob_label.pack(pady=10)
        
        self.predict_btn = tk.Button(self, text="پیش‌بینی", bg="#28a745", fg="white",
                                     font=("Arial", 14, "bold"), relief="flat", command=self.run_prediction)
        self.predict_btn.pack(pady=20, ipadx=30, ipady=10)
        
        self.canvas_frame = tk.Frame(self)
        self.canvas_frame.pack(pady=10, fill="both", expand=True)

    def run_prediction(self):
        if self.df.empty: return

        product = self.product_combo.get()
        country = self.country_combo.get()
        df_filtered = self.df[(self.df['PRODUCTLINE']==product) & (self.df['COUNTRY']==country)]
        
        if df_filtered.empty:
            messagebox.showwarning("Warning", "هیچ داده‌ای برای محصول و کشور انتخاب شده یافت نشد.")
            return

        # پاک کردن نمودار قبلی
        for widget in self.canvas_frame.winfo_children():
            widget.destroy()

        fig, ax = plt.subplots(figsize=(8,5))
        ax.plot(df_filtered['ORDERDATE'], df_filtered['SALES'].values, label="فروش واقعی")

        if lstm_model is None:
            self.prob_label.config(text="مدل LSTM بارگذاری نشده است (فقط نمایش داده‌های تاریخی)")
        else:
            preds_lstm = predict_lstm_future(df_filtered, days=30)
            if preds_lstm:
                last_sales = df_filtered['SALES'].values[-1]
                avg_pred_30 = np.mean(preds_lstm)
                
                if last_sales != 0:
                    change_pct = ((avg_pred_30 - last_sales)/last_sales)*100
                    trend_text = "افزایش" if change_pct > 0 else "کاهش"
                    self.prob_label.config(text=f"پیش‌بینی: {abs(change_pct):.2f}% {trend_text} فروش در ۳۰ روز آینده")
                
                future_x = pd.date_range(start=df_filtered['ORDERDATE'].max() + pd.Timedelta(days=1), periods=30, freq='D')
                ax.plot(future_x, preds_lstm, '--', color='red', label="پیش‌بینی (LSTM)")
            else:
                self.prob_label.config(text="داده کافی برای پیش‌بینی وجود ندارد.")

        ax.set_xlabel("زمان")
        ax.set_ylabel("فروش")
        ax.legend()
        
        canvas = FigureCanvasTkAgg(fig, master=self.canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        plt.close(fig) # مهم: بستن فیگور برای جلوگیری از پر شدن حافظه


class TopSalesFrame(tk.Frame):
    def __init__(self, master, df, **kwargs):
        super().__init__(master, **kwargs)
        self.df = df
        tk.Label(self, text="بیشترین فروش محصولات", font=("Arial", 20, "bold")).pack(pady=30)
        
        center_frame = tk.Frame(self)
        center_frame.pack(pady=20)
        tk.Label(center_frame, text="محصول:", font=("Arial", 14)).grid(row=0, column=0, padx=10, pady=10)
        self.product_combo = ttk.Combobox(center_frame, values=list(df['PRODUCTLINE'].unique()) if not df.empty else [], font=("Arial", 12), width=25)
        self.product_combo.grid(row=0, column=1, padx=10, pady=10)
        if self.product_combo['values']:
            self.product_combo.current(0)
            
        self.show_btn = tk.Button(self, text="نمایش", bg="#28a745", fg="white", font=("Arial", 14, "bold"),
                                  relief="flat", command=self.show_top_sales)
        self.show_btn.pack(pady=20, ipadx=30, ipady=10)
        
        self.result_frame = tk.Frame(self)
        self.result_frame.pack(pady=10, fill="x")
        self.line1 = tk.Label(self.result_frame, text="", font=("Arial", 14, "bold"), fg="black", justify="center")
        self.line1.pack()
        self.line2 = tk.Label(self.result_frame, text="", font=("Arial", 14, "bold"), fg="green", justify="center")
        self.line2.pack()
        self.line3 = tk.Label(self.result_frame, text="", font=("Arial", 14, "bold"), fg="black", justify="center")
        self.line3.pack()
        
        self.canvas_frame = tk.Frame(self)
        self.canvas_frame.pack(pady=10, fill="both", expand=True)

    def show_top_sales(self):
        if self.df.empty: return
        product = self.product_combo.get()
        df_filtered = self.df[self.df['PRODUCTLINE'] == product]
        
        if df_filtered.empty:
            messagebox.showwarning("Warning", "هیچ داده‌ای یافت نشد.")
            return
            
        country_sales = df_filtered.groupby('COUNTRY')['SALES'].sum().sort_values(ascending=False)
        top_country = country_sales.idxmax()
        top_sales = country_sales.max()
        start_year = df_filtered['YEAR_ID'].min()
        end_year = df_filtered['YEAR_ID'].max()
        
        self.line1.config(text=f"{product} بیشترین فروش را در {top_country} داشته است.")
        self.line2.config(text=f"مقدار فروش: {top_sales:,.0f}")
        self.line3.config(text=f"بازه سال‌ها: {start_year} تا {end_year}")
        
        for widget in self.canvas_frame.winfo_children():
            widget.destroy()
            
        fig, ax = plt.subplots(figsize=(12,6))
        country_sales.plot(kind='bar', ax=ax, color="#28a745")
        ax.set_title(f"Sales of {product} by Country")
        ax.set_xlabel("Country")
        ax.set_ylabel("Sales")
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        canvas = FigureCanvasTkAgg(fig, master=self.canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        plt.close(fig)


class CustomerOrdersFrame(tk.Frame):
    def __init__(self, master, df, **kwargs):
        super().__init__(master, **kwargs)
        self.df = df
        tk.Label(self, text="سفارشات مشتریان", font=("Arial", 20, "bold")).pack(pady=20)
        
        top_frame = tk.Frame(self)
        top_frame.pack(pady=5, fill="x")
        tk.Label(top_frame, text="انتخاب مشتری:", font=("Arial", 14)).pack(side="left", padx=10)
        
        customers = sorted(list(df["CUSTOMERNAME"].dropna().unique())) if not df.empty else []
        self.customer_box = ttk.Combobox(top_frame, values=customers, font=("Arial", 12), width=40, state="readonly")
        self.customer_box.pack(side="left", padx=10)
        if customers:
            self.customer_box.current(0)
            
        load_btn = tk.Button(top_frame, text="Load Orders", bg="#3498db", fg="white", font=("Arial", 11, "bold"),
                             relief="flat", command=self.load_customer_orders)
        load_btn.pack(side="left", padx=10)
        
        columns = ("order", "date", "sales", "country", "product", "status")
        self.tree = ttk.Treeview(self, columns=columns, show="headings", height=12)
        self.tree.pack(padx=10, pady=15, fill="both", expand=False)
        
        headers = ["شماره سفارش", "تاریخ", "فروش", "کشور", "محصول", "وضعیت"]
        for col, head in zip(columns, headers):
            self.tree.heading(col, text=head)
            self.tree.column(col, anchor="center" if col != "product" else "w")

        self.info_label = tk.Label(self, text="", font=("Arial", 13), fg="black", justify="center")
        self.info_label.pack(pady=10)

    def load_customer_orders(self):
        if self.df.empty: return
        customer = self.customer_box.get()
        data = self.df[self.df["CUSTOMERNAME"] == customer]
        
        for r in self.tree.get_children():
            self.tree.delete(r)
            
        if data.empty:
            messagebox.showinfo("Info", "هیچ سفارشی یافت نشد.")
            self.info_label.config(text="")
            return
            
        data_sorted = data.sort_values("ORDERDATE", ascending=False)
        for _, r in data_sorted.iterrows():
            date_str = r["ORDERDATE"].strftime("%Y-%m-%d") if pd.notna(r["ORDERDATE"]) else ""
            self.tree.insert("", "end", values=(
                r.get("ORDERNUMBER", ""),
                date_str,
                f"{r.get('SALES', 0):,.0f}",
                r.get("COUNTRY", ""),
                r.get("PRODUCTLINE", ""),
                r.get("STATUS", "")
            ))
            
        total_orders = len(data)
        total_sales = data["SALES"].sum()
        years = data["ORDERDATE"].dt.year
        period = f"{years.min()} تا {years.max()}" if not years.isnull().all() else "-"
        
        info_text = (
            f'مشتری "{customer}"\n'
            f'تعداد سفارش: {total_orders} | مجموع خرید: {total_sales:,.0f}$\n'
            f'دوره فعالیت: {period}'
        )
        self.info_label.config(text=info_text)


class CustomersFrame(tk.Frame):
    def __init__(self, master, df, **kwargs):
        super().__init__(master, **kwargs)
        self.df = df.copy()
        tk.Label(self, text="تحلیل کلی مشتریان", font=("Arial", 22, "bold")).pack(pady=12)

        if self.df.empty:
            tk.Label(self, text="داده‌ای موجود نیست").pack()
            return

        kpi_frame = tk.Frame(self)
        kpi_frame.pack(pady=6, fill="x", padx=12)

        total_customers = self.df['CUSTOMERNAME'].nunique()
        self.add_kpi(kpi_frame, "کل مشتریان", total_customers)

        sales_by_customer = self.df.groupby('CUSTOMERNAME')['SALES'].sum()
        avg_sales = sales_by_customer.mean() if not sales_by_customer.empty else 0
        self.add_kpi(kpi_frame, "میانگین خرید", f"{avg_sales:,.0f}$")

        orders_count = self.df.groupby('CUSTOMERNAME')['ORDERNUMBER'].nunique()
        loyal_count = (orders_count >= 3).sum()
        self.add_kpi(kpi_frame, "مشتریان وفادار", loyal_count)

        # Charts
        charts_frame = tk.Frame(self)
        charts_frame.pack(fill="both", expand=True, padx=10, pady=8)
        
        # Left Side (Top 10 Table + Chart)
        left = tk.Frame(charts_frame)
        left.pack(side="left", fill="both", expand=True, padx=6)
        
        tk.Label(left, text="۱۰ مشتری برتر (بر اساس فروش)", font=("Arial", 12, "bold")).pack(pady=6)
        
        self.top_tree = ttk.Treeview(left, columns=("customer", "sales"), show="headings", height=6)
        self.top_tree.heading("customer", text="مشتری")
        self.top_tree.heading("sales", text="فروش ($)")
        self.top_tree.pack(pady=6, fill="x")
        
        self.top10_canvas = tk.Frame(left)
        self.top10_canvas.pack(fill="both", expand=True)

        # Right Side (Country + Growth)
        right = tk.Frame(charts_frame)
        right.pack(side="left", fill="both", expand=True, padx=6)
        
        tk.Label(right, text="توزیع فروش بر اساس کشور", font=("Arial", 12, "bold")).pack(pady=6)
        self.country_canvas = tk.Frame(right)
        self.country_canvas.pack(fill="both", expand=True)
        
        tk.Label(right, text="رشد مشتریان (ماهانه)", font=("Arial", 12, "bold")).pack(pady=6)
        self.growth_canvas = tk.Frame(right)
        self.growth_canvas.pack(fill="both", expand=True)

        self.render_top10()
        self.render_country_dist()
        self.render_growth()

    def add_kpi(self, parent, title, value):
        lbl = tk.Label(parent, text=f"{title}\n{value}", font=("Arial", 12, "bold"), bd=1, relief="solid", padx=12, pady=8)
        lbl.pack(side="left", padx=8, ipadx=6, fill="y")

    def render_top10(self):
        sales_by_customer = self.df.groupby('CUSTOMERNAME')['SALES'].sum().sort_values(ascending=False)
        top10 = sales_by_customer.head(10)
        
        for r in self.top_tree.get_children():
            self.top_tree.delete(r)
        for cust, val in top10.items():
            self.top_tree.insert("", "end", values=(cust, f"{val:,.0f}"))

        fig, ax = plt.subplots(figsize=(6,3.5))
        top10[::-1].plot(kind='barh', ax=ax, color='#2c7fb8')
        ax.set_xlabel("فروش")
        plt.tight_layout()
        self.draw_chart(fig, self.top10_canvas)

    def render_country_dist(self):
        sales_by_country = self.df.groupby('COUNTRY')['SALES'].sum().sort_values(ascending=False).head(10)
        fig, ax = plt.subplots(figsize=(6,3.5))
        sales_by_country.plot(kind='bar', ax=ax, color='#91cf60')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        plt.tight_layout()
        self.draw_chart(fig, self.country_canvas)

    def render_growth(self):
        tmp = self.df.copy()
        tmp['year_month'] = tmp['ORDERDATE'].dt.to_period('M').dt.to_timestamp()
        customers_month = tmp.groupby('year_month')['CUSTOMERNAME'].nunique()
        fig, ax = plt.subplots(figsize=(6,3.5))
        ax.plot(customers_month.index, customers_month.values, marker='o')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        fig.autofmt_xdate()
        plt.tight_layout()
        self.draw_chart(fig, self.growth_canvas)

    def draw_chart(self, fig, parent):
        for widget in parent.winfo_children():
            widget.destroy()
        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        plt.close(fig)


class CustomerInfoFrame(tk.Frame):
    def __init__(self, master, df, **kwargs):
        super().__init__(master, **kwargs)
        self.df = df.copy()
        tk.Label(self, text="اطلاعات مشتریان", font=("Arial", 22, "bold")).pack(pady=12)

        top_frame = tk.Frame(self)
        top_frame.pack(pady=10, fill="x")
        
        tk.Label(top_frame, text="انتخاب نوع مشتری:", font=("Arial", 14)).pack(side="left", padx=10)
        self.type_combo = ttk.Combobox(top_frame, values=["کل مشتریان", "وفاداران"], font=("Arial", 12), width=20, state="readonly")
        self.type_combo.pack(side="left", padx=10)
        self.type_combo.current(0)
        
        load_btn = tk.Button(top_frame, text="نمایش", bg="#3498db", fg="white", font=("Arial", 11, "bold"),
                             relief="flat", command=self.load_customers)
        load_btn.pack(side="left", padx=10)

        self.tree = ttk.Treeview(self, columns=("name", "phone", "country"), show="headings", height=20)
        self.tree.pack(padx=10, pady=15, fill="both", expand=True)
        self.tree.heading("name", text="نام مشتری")
        self.tree.heading("phone", text="تلفن")
        self.tree.heading("country", text="کشور")

    def load_customers(self):
        if self.df.empty: return
        selection = self.type_combo.get()
        
        cols = ["CUSTOMERNAME", "PHONE", "COUNTRY"]
        # بررسی وجود ستون‌ها
        cols = [c for c in cols if c in self.df.columns]
        
        if selection == "کل مشتریان":
            data = self.df[cols].drop_duplicates()
        elif selection == "وفاداران":
            orders_count = self.df.groupby("CUSTOMERNAME")["ORDERNUMBER"].nunique()
            loyal_customers = orders_count[orders_count >= 3].index
            data = self.df[self.df["CUSTOMERNAME"].isin(loyal_customers)][cols].drop_duplicates()
        else:
            data = pd.DataFrame(columns=cols)

        for r in self.tree.get_children():
            self.tree.delete(r)
        
        for _, row in data.iterrows():
            vals = [row.get(c, "") for c in ["CUSTOMERNAME", "PHONE", "COUNTRY"]]
            self.tree.insert("", "end", values=vals)


class ProductCustomerFrame(tk.Frame):
    def __init__(self, master, df, **kwargs):
        super().__init__(master, **kwargs)
        self.df = df.copy()
        tk.Label(self, text="هر محصول برای کدام مشتری؟", font=("Arial", 22, "bold")).pack(pady=12)

        top = tk.Frame(self)
        top.pack(pady=5)
        tk.Label(top, text="انتخاب محصول:", font=("Arial", 14)).pack(side="left", padx=10)
        
        products = sorted(self.df["PRODUCTCODE"].unique()) if not self.df.empty else []
        self.cmb = ttk.Combobox(top, values=products, width=25, state="readonly", font=("Arial", 12))
        self.cmb.pack(side="left")
        self.cmb.bind("<<ComboboxSelected>>", self.analyze)

        columns = ("customer", "count", "sales", "score", "percent")
        self.tree = ttk.Treeview(self, columns=columns, show="headings", height=15)
        self.tree.pack(fill="both", expand=True, pady=15)
        headers = ["مشتری", "تعداد خرید", "کل فروش", "امتیاز", "احتمال خرید ٪"]
        for c, h in zip(columns, headers):
            self.tree.heading(c, text=h)
            self.tree.column(c, anchor="center")

        self.result_lbl = tk.Label(self, text="", font=("Arial", 16, "bold"), fg="green")
        self.result_lbl.pack(pady=10)

    def analyze(self, event=None):
        if self.df.empty: return
        product = self.cmb.get()
        df_p = self.df[self.df["PRODUCTCODE"] == product]

        customer_counts = df_p.groupby("CUSTOMERNAME")["ORDERNUMBER"].nunique()
        customer_sales = df_p.groupby("CUSTOMERNAME")["SALES"].sum()

        similar = self.df[self.df["PRODUCTLINE"] == df_p["PRODUCTLINE"].iloc[0]]
        similar_counts = similar.groupby("CUSTOMERNAME")["ORDERNUMBER"].nunique()

        max_count = customer_counts.max() if not customer_counts.empty else 1
        max_sales = customer_sales.max() if not customer_sales.empty else 1
        max_similar = similar_counts.max() if not similar_counts.empty else 1

        results = []
        for c in customer_counts.index:
            score = (
                (customer_counts[c] / max_count) * 0.5 +
                (customer_sales[c] / max_sales) * 0.3 +
                (similar_counts.get(c, 0) / max_similar) * 0.2
            )
            results.append([c, customer_counts[c], f"{customer_sales[c]:,.0f}", round(score, 3), int(score * 100)])

        for i in self.tree.get_children():
            self.tree.delete(i)

        avg_score = 0
        if results:
            # Sort by score
            results.sort(key=lambda x: x[3], reverse=True)
            for r in results:
                self.tree.insert("", "end", values=r)
            avg_score = sum([r[3] for r in results]) / len(results)

        # محاسبه ساده ارزش تولید
        expected_value = avg_score * df_p["QUANTITYORDERED"].mean() * df_p["PRICEEACH"].mean()
        production_cost = df_p["PRICEEACH"].mean() * 0.6 # فرض: هزینه تولید ۶۰ درصد قیمت است
        worth = (expected_value / production_cost) * 100 if production_cost > 0 else 0
        
        self.result_lbl.config(text=f"ارزش تخمینی تولید مجدد: {worth:.1f}%")


class AnalyticsFrame(tk.Frame):
    def __init__(self, parent, df):
        super().__init__(parent)
        self.df = df.copy()
        
        canvas = tk.Canvas(self)
        scrollbar = tk.Scrollbar(self, orient="vertical", command=canvas.yview)
        scroll_frame = tk.Frame(canvas)

        scroll_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        self.build_charts(scroll_frame)

    def build_charts(self, container):
        if self.df.empty: return
        sns.set(style="whitegrid")

        def add_chart(title, fig):
            lbl = tk.Label(container, text=title, font=("Arial", 16, "bold"))
            lbl.pack(pady=10)
            canvas = FigureCanvasTkAgg(fig, master=container)
            canvas.get_tk_widget().pack(pady=10)
            canvas.draw()
            plt.close(fig) # بستن فیگور

        # 1) Top Products
        fig1 = plt.figure(figsize=(8, 4))
        top_prod = self.df.groupby("PRODUCTLINE")["SALES"].sum().sort_values(ascending=False).head(10)
        sns.barplot(x=top_prod.values, y=top_prod.index, palette="viridis")
        plt.title("محصولات پرفروش")
        add_chart("پرفروش‌ترین محصولات", fig1)

        # 2) Monthly Trend
        fig2 = plt.figure(figsize=(8, 4))
        trend = self.df.groupby("MONTH_ID")["SALES"].sum().sort_index()
        sns.lineplot(x=trend.index, y=trend.values, marker="o")
        plt.title("روند فروش ماهانه")
        add_chart("روند فروش ماهیانه", fig2)

        # 3) Pie Chart
        fig3 = plt.figure(figsize=(6, 6))
        share = self.df.groupby("PRODUCTLINE")["SALES"].sum()
        plt.pie(share, labels=share.index, autopct="%1.1f%%")
        plt.title("سهم بازار")
        add_chart("سهم هر محصول از بازار", fig3)
        
        # 4) Heatmap
        fig6 = plt.figure(figsize=(8, 6))
        num_df = self.df.select_dtypes(include=['int', 'float'])
        sns.heatmap(num_df.corr(), annot=False, cmap="coolwarm")
        plt.title("نقشه همبستگی متغیرها")
        add_chart("هیت‌مپ همبستگی", fig6)


class CRMApp:
    def __init__(self, master):
        self.master = master
        master.title("سیستم CRM هوشمند")
        master.geometry("1300x800")

        # Sidebar
        self.sidebar = tk.Frame(master, width=280, bg="#2c3e50")
        self.sidebar.pack(side="left", fill="y")
        tk.Label(self.sidebar, text="منوی ابزارها", bg="#2c3e50", fg="white", font=("Arial", 16, "bold")).pack(pady=20)

        menu_items = [
            "HOME",
            "پیش‌بینی آینده فروش محصول",
            "بیشترین فروش محصول",
            "سفارشات مشتریان",
            "تحلیل کلی مشتریان",
            "اطلاعات مشتریان",
            "هر محصول برای کدام مشتری؟",
            "نمودارهای تحلیلی"
        ]

        for item in menu_items:
            btn = tk.Button(self.sidebar, text=item, font=("Arial", 12), bg="#34495e", fg="white",
                            relief="flat", command=lambda n=item: self.show_content(n), height=2)
            btn.pack(fill="x", pady=2, padx=10)

        # Main Content Area
        self.main_frame = tk.Frame(master, bg="#ecf0f1")
        self.main_frame.pack(side="left", fill="both", expand=True)

        self.current_widget = None
        self.show_home()

    def clear_main(self):
        for w in self.main_frame.winfo_children():
            w.destroy()

    def show_home(self):
        self.clear_main()
        tk.Label(self.main_frame, text="به سیستم CRM خوش آمدید", font=("Arial", 30, "bold"), bg="#ecf0f1").pack(pady=50)
        
        status_text = "وضعیت سیستم:\n\n"
        status_text += f"✅ دیتاست بارگذاری شد: {len(df)} ردیف\n" if not df.empty else "❌ دیتاست یافت نشد\n"
        status_text += "✅ مدل MLP متصل است\n" if mlp_model else "❌ مدل MLP یافت نشد\n"
        status_text += "✅ مدل LSTM متصل است\n" if lstm_model else "❌ مدل LSTM یافت نشد\n"
        
        tk.Label(self.main_frame, text=status_text, font=("Arial", 14), bg="#ecf0f1", justify="left").pack(pady=20)

    def show_content(self, name):
        self.clear_main()
        
        if name == "HOME":
            self.show_home()
            return
            
        # دیکشنری نگاشت نام منو به کلاس‌ها
        frames = {
            "پیش‌بینی آینده فروش محصول": FutureSalesFrame,
            "بیشترین فروش محصول": TopSalesFrame,
            "سفارشات مشتریان": CustomerOrdersFrame,
            "تحلیل کلی مشتریان": CustomersFrame,
            "اطلاعات مشتریان": CustomerInfoFrame,
            "هر محصول برای کدام مشتری؟": ProductCustomerFrame,
            "نمودارهای تحلیلی": AnalyticsFrame
        }
        
        if name in frames:
            # ایجاد نمونه از کلاس مربوطه
            widget = frames[name](self.main_frame, df)
            widget.pack(fill="both", expand=True)
        else:
            tk.Label(self.main_frame, text="بخش در حال توسعه...", font=("Arial", 20)).pack(pady=50)

if __name__ == "__main__":
    root = tk.Tk()
    # تنظیم استایل کلی
    style = ttk.Style()
    style.theme_use('clam')
    app = CRMApp(root)
    root.mainloop()

# import tkinter as tk
# from tkinter import ttk, messagebox
# import pandas as pd
# import numpy as np
# from pathlib import Path
# import joblib
# from tensorflow.keras.models import load_model
# from tensorflow.keras.losses import MeanSquaredError
# import matplotlib.pyplot as plt
# from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
# import matplotlib.dates as mdates

# mlp_model_path = Path(r"D:\deeplearning\project\CRM2\models\crm_sales_model_final.h5")
# lstm_model_path = Path(r"D:\deeplearning\project\CRM2\models\lstm_sales_model_final.h5")
# preprocessor_path = Path(r"D:\deeplearning\project\CRM2\models\preprocessor.pkl")
# dataset_path = Path(r"D:\deeplearning\project\CRM2\sales_data_sample.csv")
# churn_dataset_path = Path(r"D:\deeplearning\project\CRM2\sales_data_sample.csv")
# churn_df = pd.read_csv(churn_dataset_path)

# df = pd.read_csv(dataset_path, encoding='cp1252')
# df['ORDERDATE'] = pd.to_datetime(df['ORDERDATE'])
# if 'YEAR_ID' not in df.columns:
#     df['YEAR_ID'] = df['ORDERDATE'].dt.year
# if 'MONTH_ID' not in df.columns:
#     df['MONTH_ID'] = df['ORDERDATE'].dt.month

# mlp_model = None
# lstm_model = None
# preprocessor = None
# try:
#     mlp_model = load_model(mlp_model_path, custom_objects={'mse': MeanSquaredError()})
# except Exception:
#     mlp_model = None

# try:
#     lstm_model = load_model(lstm_model_path, compile=False)
#     lstm_model.compile(optimizer='adam', loss='mse', metrics=['mse'])
# except Exception:
#     lstm_model = None

# try:
#     preprocessor = joblib.load(preprocessor_path)
# except Exception:
#     preprocessor = None

# def preprocess_mlp(df_):
#     features = ['QUANTITYORDERED', 'PRICEEACH', 'ORDERLINENUMBER',
#                 'PRODUCTLINE', 'MSRP', 'PRODUCTCODE', 'COUNTRY', 'TERRITORY',
#                 'MONTH_ID', 'YEAR_ID']
#     X = df_[features]
#     X_processed = preprocessor.transform(X)
#     return X_processed

# def predict_mlp(df_):
#     if mlp_model is None or preprocessor is None:
#         raise RuntimeError("MLP model or preprocessor not loaded.")
#     X = preprocess_mlp(df_)
#     preds = mlp_model.predict(X, verbose=0)
#     return preds

# def predict_lstm_future(df_, days=30, seq_length=30):
#     if lstm_model is None:
#         raise RuntimeError("LSTM model not loaded.")
#     series = df_['SALES'].values
#     history = series[-seq_length:]
#     preds = []
#     for _ in range(days):
#         X_input = np.array(history[-seq_length:]).reshape((1, seq_length,1))
#         pred = lstm_model.predict(X_input, verbose=0)[0][0]
#         preds.append(pred)
#         history = np.append(history, pred)
#     return preds



# def preprocess_churn(df):
#     df = df.copy()
#     le = LabelEncoder()
#     for col in df.select_dtypes(include=['object']).columns:
#         df[col] = le.fit_transform(df[col])
#     df.fillna(0, inplace=True)
#     X = df.drop(columns=['Churn'])
#     y = df['Churn'].values
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X)
#     return X_scaled, y, scaler

# X, y, scaler = preprocess_churn(churn_df)


# class FutureSalesFrame(tk.Frame):
#     def __init__(self, master, df, **kwargs):
#         super().__init__(master, **kwargs)
#         self.df = df
#         tk.Label(self, text="پیش‌بینی آینده فروش محصول", font=("Arial", 20, "bold")).pack(pady=30)
#         center_frame = tk.Frame(self)
#         center_frame.pack(pady=20)
#         tk.Label(center_frame, text="محصول:", font=("Arial", 14)).grid(row=0, column=0, padx=10, pady=10)
#         self.product_combo = ttk.Combobox(center_frame, values=list(df['PRODUCTLINE'].unique()), font=("Arial", 12), width=25)
#         self.product_combo.grid(row=0, column=1, padx=10, pady=10)
#         if len(self.product_combo['values'])>0:
#             self.product_combo.current(0)
#         tk.Label(center_frame, text="کشور:", font=("Arial", 14)).grid(row=1, column=0, padx=10, pady=10)
#         self.country_combo = ttk.Combobox(center_frame, values=list(df['COUNTRY'].unique()), font=("Arial", 12), width=25)
#         self.country_combo.grid(row=1, column=1, padx=10, pady=10)
#         if len(self.country_combo['values'])>0:
#             self.country_combo.current(0)
#         self.prob_label = tk.Label(self, text="به احتمال ... درصد در ۳۰ روز آینده فروش خواهد داشت", font=("Arial", 14), fg="blue")
#         self.prob_label.pack(pady=10)
#         self.predict_btn = tk.Button(self, text="پیش‌بینی", bg="#28a745", fg="white",
#                                      font=("Arial", 14, "bold"), relief="flat", command=self.run_prediction)
#         self.predict_btn.pack(pady=20, ipadx=30, ipady=10)
#         self.canvas_frame = tk.Frame(self)
#         self.canvas_frame.pack(pady=10, fill="both", expand=True)

#     def run_prediction(self):
#         product = self.product_combo.get()
#         country = self.country_combo.get()
#         df_filtered = self.df[(self.df['PRODUCTLINE']==product) & (self.df['COUNTRY']==country)]
#         if df_filtered.empty:
#             messagebox.showwarning("Warning", "هیچ داده‌ای برای محصول و کشور انتخاب شده یافت نشد.")
#             return

#         if lstm_model is None:
#             messagebox.showinfo("Info", "LSTM model not available — only plotting historical sales.")
#             fig, ax = plt.subplots(figsize=(8,5))
#             ax.plot(df_filtered['ORDERDATE'], df_filtered['SALES'].values, label="real sell")
#             ax.set_xlabel("Time")
#             ax.set_ylabel("Sales")
#             ax.legend()
#         else:
#             preds_lstm = predict_lstm_future(df_filtered, days=30)
#             last_sales = df_filtered['SALES'].values[-1]
#             avg_pred_30 = np.mean(preds_lstm)
#             prob_30 = ((avg_pred_30 - last_sales)/last_sales)*100
#             self.prob_label.config(text=f"به احتمال {prob_30:.2f}% در ۳۰ روز آینده فروش خواهد داشت")
#             fig, ax = plt.subplots(figsize=(8,5))
#             ax.plot(df_filtered['ORDERDATE'], df_filtered['SALES'].values, label="real sell")
#             future_x = pd.date_range(start=df_filtered['ORDERDATE'].max() + pd.Timedelta(days=1), periods=30, freq='D')
#             ax.plot(future_x, preds_lstm, '--', label="Next 30 days(LSTM)")
#             ax.set_xlabel("Time")
#             ax.set_ylabel("Sales")
#             ax.legend()

#         for widget in self.canvas_frame.winfo_children():
#             widget.destroy()
#         canvas = FigureCanvasTkAgg(fig, master=self.canvas_frame)
#         canvas.draw()
#         canvas.get_tk_widget().pack(fill="both", expand=True)


# class TopSalesFrame(tk.Frame):
#     def __init__(self, master, df, **kwargs):
#         super().__init__(master, **kwargs)
#         self.df = df
#         tk.Label(self, text="بیشترین فروش محصولات", font=("Arial", 20, "bold")).pack(pady=30)
#         center_frame = tk.Frame(self)
#         center_frame.pack(pady=20)
#         tk.Label(center_frame, text="محصول:", font=("Arial", 14)).grid(row=0, column=0, padx=10, pady=10)
#         self.product_combo = ttk.Combobox(center_frame, values=list(df['PRODUCTLINE'].unique()), font=("Arial", 12), width=25)
#         self.product_combo.grid(row=0, column=1, padx=10, pady=10)
#         if len(self.product_combo['values'])>0:
#             self.product_combo.current(0)
#         self.show_btn = tk.Button(self, text="نمایش", bg="#28a745", fg="white", font=("Arial", 14, "bold"),
#                                   relief="flat", command=self.show_top_sales)
#         self.show_btn.pack(pady=20, ipadx=30, ipady=10)
#         self.result_frame = tk.Frame(self)
#         self.result_frame.pack(pady=10, fill="x")
#         self.line1 = tk.Label(self.result_frame, text="", font=("Arial", 14, "bold"), fg="black", justify="center")
#         self.line1.pack()
#         self.line2 = tk.Label(self.result_frame, text="", font=("Arial", 14, "bold"), fg="green", justify="center")
#         self.line2.pack()
#         self.line3 = tk.Label(self.result_frame, text="", font=("Arial", 14, "bold"), fg="black", justify="center")
#         self.line3.pack()
#         self.canvas_frame = tk.Frame(self)
#         self.canvas_frame.pack(pady=10, fill="both", expand=True)

#     def show_top_sales(self):
#         product = self.product_combo.get()
#         df_filtered = self.df[self.df['PRODUCTLINE'] == product]
#         if df_filtered.empty:
#             messagebox.showwarning("Warning", "هیچ داده‌ای برای محصول انتخاب شده یافت نشد.")
#             return
#         country_sales = df_filtered.groupby('COUNTRY')['SALES'].sum().sort_values(ascending=False)
#         top_country = country_sales.idxmax()
#         top_sales = country_sales.max()
#         start_year = df_filtered['YEAR_ID'].min()
#         end_year = df_filtered['YEAR_ID'].max()
#         self.line1.config(text=f"{product} بیشترین فروش را در {top_country} داشته است.")
#         self.line2.config(text=f"مقدار فروش: {top_sales:,}")
#         self.line3.config(text=f"بازه سال‌ها: {start_year} تا {end_year}")
#         fig, ax = plt.subplots(figsize=(12,6))

#         country_sales.plot(kind='bar', ax=ax, color="#28a745")
#         ax.set_title(f"Sales of {product} by Country")
#         ax.set_xlabel("Country")
#         ax.set_ylabel("Sales")
#         plt.xticks(rotation=45)
#         for widget in self.canvas_frame.winfo_children():
#             widget.destroy()
#         canvas = FigureCanvasTkAgg(fig, master=self.canvas_frame)
#         canvas.draw()
#         canvas.get_tk_widget().pack(fill="both", expand=True)


# class CustomerOrdersFrame(tk.Frame):
#     def __init__(self, master, df, **kwargs):
#         super().__init__(master, **kwargs)
#         self.df = df
#         tk.Label(self, text="سفارشات مشتریان", font=("Arial", 20, "bold")).pack(pady=20)
#         top_frame = tk.Frame(self)
#         top_frame.pack(pady=5, fill="x")
#         tk.Label(top_frame, text="انتخاب مشتری:", font=("Arial", 14)).pack(side="left", padx=10)
#         customers = sorted(list(df["CUSTOMERNAME"].dropna().unique()))
#         self.customer_box = ttk.Combobox(top_frame, values=customers, font=("Arial", 12), width=40, state="readonly")
#         self.customer_box.pack(side="left", padx=10)
#         if customers:
#             self.customer_box.current(0)
#         load_btn = tk.Button(top_frame, text="Load Orders", bg="#3498db", fg="white", font=("Arial", 11, "bold"),
#                              relief="flat", command=self.load_customer_orders)
#         load_btn.pack(side="left", padx=10)
#         columns = ("order", "date", "sales", "country", "product", "status")
#         self.tree = ttk.Treeview(self, columns=columns, show="headings", height=12)
#         self.tree.pack(padx=10, pady=15, fill="both", expand=False)
#         self.tree.heading("order", text="Order No")
#         self.tree.heading("date", text="Order Date")
#         self.tree.heading("sales", text="Sales")
#         self.tree.heading("country", text="Country")
#         self.tree.heading("product", text="Product")
#         self.tree.heading("status", text="Status")
#         self.tree.column("order", width=100, anchor="center")
#         self.tree.column("date", width=130, anchor="center")
#         self.tree.column("sales", width=120, anchor="e")
#         self.tree.column("country", width=120, anchor="center")
#         self.tree.column("product", width=220, anchor="w")
#         self.tree.column("status", width=120, anchor="center")
#         self.info_label = tk.Label(self, text="", font=("Arial", 13), fg="black", justify="center")
#         self.info_label.pack(pady=10)

#     def load_customer_orders(self):
#         customer = self.customer_box.get()
#         data = self.df[self.df["CUSTOMERNAME"] == customer]
#         for r in self.tree.get_children():
#             self.tree.delete(r)
#         if data.empty:
#             messagebox.showinfo("Info", "هیچ سفارشی برای این مشتری وجود ندارد.")
#             self.info_label.config(text="")
#             return
#         data_sorted = data.sort_values("ORDERDATE", ascending=False)
#         for _, r in data_sorted.iterrows():
#             order_no = r.get("ORDERNUMBER", "")
#             date = r.get("ORDERDATE", "")
#             sales = r.get("SALES", 0)
#             country = r.get("COUNTRY", "")
#             product = r.get("PRODUCTLINE", "")
#             status = r.get("STATUS", "")
#             date_str = date.strftime("%Y-%m-%d") if not pd.isna(date) else ""
#             self.tree.insert("", "end", values=(order_no, date_str, f"{sales:,.0f}", country, product, status))
#         total_orders = len(data)
#         total_sales = data["SALES"].sum()
#         years = pd.to_datetime(data["ORDERDATE"]).dt.year
#         start_year = int(years.min()) if not years.isna().all() else ""
#         end_year = int(years.max()) if not years.isna().all() else ""
#         info_text = (
#             f'مشتری "{customer}" تا کنون {total_orders} سفارش ثبت کرده است.\n'
#             f'مجموع فروش: {total_sales:,.0f}$\n'
#             f'بازه فعالیت: {start_year} تا {end_year}'
#         )
#         self.info_label.config(text=info_text)


# class CustomersFrame(tk.Frame):
#     """بخش مشتریان: KPI، Top10، country distribution، growth"""
#     def __init__(self, master, df, **kwargs):
#         super().__init__(master, **kwargs)
#         self.df = df.copy()
#         tk.Label(self, text="تحلیل کلی مشتریان", font=("Arial", 22, "bold")).pack(pady=12)

#         kpi_frame = tk.Frame(self)
#         kpi_frame.pack(pady=6, fill="x", padx=12)

#         total_customers = self.df['CUSTOMERNAME'].nunique()
#         self.k1 = tk.Label(kpi_frame, text=f"Total Customers\n{total_customers}", font=("Arial", 12, "bold"), bd=1, relief="solid", padx=12, pady=8)
#         self.k1.pack(side="left", padx=8, ipadx=6, fill="y")

#         sales_by_customer = self.df.groupby('CUSTOMERNAME')['SALES'].sum()
#         avg_sales = sales_by_customer.mean() if not sales_by_customer.empty else 0
#         self.k2 = tk.Label(kpi_frame, text=f"Avg Sales / Customer\n{avg_sales:,.0f}$", font=("Arial", 12, "bold"), bd=1, relief="solid", padx=12, pady=8)
#         self.k2.pack(side="left", padx=8, ipadx=6, fill="y")

#         #  loyal customers (>=3 orders)
#         orders_count = self.df.groupby('CUSTOMERNAME')['ORDERNUMBER'].nunique()
#         loyal_count = (orders_count >= 3).sum()
#         self.k3 = tk.Label(kpi_frame, text=f"Loyal Customers\n{loyal_count}", font=("Arial", 12, "bold"), bd=1, relief="solid", padx=12, pady=8)
#         self.k3.pack(side="left", padx=8, ipadx=6, fill="y")

#         charts_frame = tk.Frame(self)
#         charts_frame.pack(fill="both", expand=True, padx=10, pady=8)

#         left = tk.Frame(charts_frame)
#         left.pack(side="left", fill="both", expand=True, padx=6)

#         tk.Label(left, text="Top 10 Customers (by Sales)", font=("Arial", 12, "bold")).pack(pady=6)
#         self.top10_canvas_frame = tk.Frame(left)
#         self.top10_canvas_frame.pack(fill="both", expand=True)

#         self.top_tree = ttk.Treeview(left, columns=("customer", "sales"), show="headings", height=6)
#         self.top_tree.heading("customer", text="Customer")
#         self.top_tree.heading("sales", text="Sales")
#         self.top_tree.column("customer", width=200)
#         self.top_tree.column("sales", width=120, anchor="e")
#         self.top_tree.pack(pady=6, fill="x")

#         right = tk.Frame(charts_frame)
#         right.pack(side="left", fill="both", expand=True, padx=6)

#         tk.Label(right, text="Sales by Country", font=("Arial", 12, "bold")).pack(pady=6)
#         self.country_canvas_frame = tk.Frame(right)
#         self.country_canvas_frame.pack(fill="both", expand=True)

#         tk.Label(right, text="Customers growth (monthly)", font=("Arial", 12, "bold")).pack(pady=6)
#         self.growth_canvas_frame = tk.Frame(right)
#         self.growth_canvas_frame.pack(fill="both", expand=True)

#         self.render_top10()
#         self.render_country_dist()
#         self.render_growth()

#     def render_top10(self):
#         sales_by_customer = self.df.groupby('CUSTOMERNAME')['SALES'].sum().sort_values(ascending=False)
#         top10 = sales_by_customer.head(10)
#         for r in self.top_tree.get_children():
#             self.top_tree.delete(r)
#         for cust, val in top10.items():
#             self.top_tree.insert("", "end", values=(cust, f"{val:,.0f}$"))

#         # plot
#         fig, ax = plt.subplots(figsize=(6,3.5))
#         top10[::-1].plot(kind='barh', ax=ax, color='#2c7fb8')
#         ax.set_xlabel("Sales")
#         ax.set_ylabel("")
#         ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
#         fig.tight_layout()
#         for widget in self.top10_canvas_frame.winfo_children():
#             widget.destroy()
#         canvas = FigureCanvasTkAgg(fig, master=self.top10_canvas_frame)
#         canvas.draw()
#         canvas.get_tk_widget().pack(fill="both", expand=True)

#     def render_country_dist(self):
#         sales_by_country = self.df.groupby('COUNTRY')['SALES'].sum().sort_values(ascending=False)
#         fig, ax = plt.subplots(figsize=(6,3.5))
#         sales_by_country.plot(kind='bar', ax=ax, color='#91cf60')
#         ax.set_xlabel("Country")
#         ax.set_ylabel("Sales")
#         ax.xaxis.set_tick_params(rotation=45)
#         ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
#         fig.tight_layout()
#         for widget in self.country_canvas_frame.winfo_children():
#             widget.destroy()
#         canvas = FigureCanvasTkAgg(fig, master=self.country_canvas_frame)
#         canvas.draw()
#         canvas.get_tk_widget().pack(fill="both", expand=True)

#     def render_growth(self):
#         tmp = self.df.copy()
#         tmp['year_month'] = tmp['ORDERDATE'].dt.to_period('M').dt.to_timestamp()
#         customers_month = tmp.groupby('year_month')['CUSTOMERNAME'].nunique()
#         fig, ax = plt.subplots(figsize=(6,3.5))
#         ax.plot(customers_month.index, customers_month.values, marker='o')
#         ax.set_xlabel("Month")
#         ax.set_ylabel("Unique Customers")
#         ax.xaxis.set_major_locator(mdates.AutoDateLocator())
#         ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
#         fig.autofmt_xdate(rotation=45)
#         fig.tight_layout()
#         for widget in self.growth_canvas_frame.winfo_children():
#             widget.destroy()
#         canvas = FigureCanvasTkAgg(fig, master=self.growth_canvas_frame)
#         canvas.draw()
#         canvas.get_tk_widget().pack(fill="both", expand=True)





# class CustomerInfoFrame(tk.Frame):
#     """نمایش اطلاعات مشتریان: کل مشتریان و وفاداران"""
#     def __init__(self, master, df, **kwargs):
#         super().__init__(master, **kwargs)
#         self.df = df.copy()

#         tk.Label(self, text="اطلاعات مشتریان", font=("Arial", 22, "bold")).pack(pady=12)

#         # فریم بالا: انتخاب نوع مشتری
#         top_frame = tk.Frame(self)
#         top_frame.pack(pady=10, fill="x")

#         tk.Label(top_frame, text="انتخاب نوع مشتری:", font=("Arial", 14)).pack(side="left", padx=10)

#         self.type_combo = ttk.Combobox(
#             top_frame,
#             values=["کل مشتریان", "وفاداران"],
#             font=("Arial", 12),
#             width=20,
#             state="readonly"
#         )
#         self.type_combo.pack(side="left", padx=10)
#         self.type_combo.current(0)

#         load_btn = tk.Button(
#             top_frame,
#             text="نمایش",
#             bg="#3498db",
#             fg="white",
#             font=("Arial", 11, "bold"),
#             relief="flat",
#             command=self.load_customers
#         )
#         load_btn.pack(side="left", padx=10)


#         columns = ("name", "phone", "country")
#         self.tree = ttk.Treeview(self, columns=columns, show="headings", height=20)
#         self.tree.pack(padx=10, pady=15, fill="both", expand=True)
#         self.tree.heading("name", text="نام مشتری")
#         self.tree.heading("phone", text="شماره تماس")
#         self.tree.heading("country", text="کشور")
#         self.tree.column("name", width=250)
#         self.tree.column("phone", width=150, anchor="center")
#         self.tree.column("country", width=120, anchor="center")

#     def load_customers(self):
#         selection = self.type_combo.get()

#         if selection == "کل مشتریان":
#             data = self.df[["CUSTOMERNAME", "PHONE", "COUNTRY"]].drop_duplicates()
#         elif selection == "وفاداران":
#             orders_count = self.df.groupby("CUSTOMERNAME")["ORDERNUMBER"].nunique()
#             loyal_customers = orders_count[orders_count >= 3].index
#             data = self.df[self.df["CUSTOMERNAME"].isin(loyal_customers)][["CUSTOMERNAME", "PHONE", "COUNTRY"]].drop_duplicates()
#         else:
#             data = pd.DataFrame(columns=["CUSTOMERNAME", "PHONE", "COUNTRY"])

#         for r in self.tree.get_children():
#             self.tree.delete(r)

#         for _, row in data.iterrows():
#             self.tree.insert("", "end", values=(row["CUSTOMERNAME"], row["PHONE"], row["COUNTRY"]))




# class ProductCustomerFrame(tk.Frame):
#     """تحلیل اینکه هر محصول را کدام مشتری احتمالاً دوباره می‌خرد،
#        و اینکه تولید مجدد آن محصول چقدر ارزش دارد."""
#     def __init__(self, master, df, **kwargs):
#         super().__init__(master, **kwargs)
#         self.df = df.copy()

#         tk.Label(self, text="هر محصول برای کدام مشتری؟", font=("Arial", 22, "bold")).pack(pady=12)

#         top = tk.Frame(self)
#         top.pack(pady=5)

#         tk.Label(top, text="انتخاب محصول:", font=("Arial", 14)).pack(side="left", padx=10)

#         products = sorted(self.df["PRODUCTCODE"].unique())
#         self.cmb = ttk.Combobox(top, values=products, width=25, state="readonly", font=("Arial", 12))
#         self.cmb.pack(side="left")
#         self.cmb.bind("<<ComboboxSelected>>", self.analyze)

#         columns = ("customer", "count", "sales", "score", "percent")
#         self.tree = ttk.Treeview(self, columns=columns, show="headings", height=18)
#         self.tree.pack(fill="both", expand=True, pady=15)

#         self.tree.heading("customer", text="مشتری")
#         self.tree.heading("count", text="تعداد خرید")
#         self.tree.heading("sales", text="کل فروش")
#         self.tree.heading("score", text="امتیاز خرید")
#         self.tree.heading("percent", text="٪ احتمال خرید")

#         self.result_lbl = tk.Label(self, text="", font=("Arial", 16, "bold"), fg="green")
#         self.result_lbl.pack(pady=10)

#     def analyze(self, event=None):
#         product = self.cmb.get()
#         df_p = self.df[self.df["PRODUCTCODE"] == product]

#         customer_counts = df_p.groupby("CUSTOMERNAME")["ORDERNUMBER"].nunique()
#         customer_sales = df_p.groupby("CUSTOMERNAME")["SALES"].sum()

#         similar = self.df[self.df["PRODUCTLINE"] == df_p["PRODUCTLINE"].iloc[0]]
#         similar_counts = similar.groupby("CUSTOMERNAME")["ORDERNUMBER"].nunique()

#         max_count = customer_counts.max()
#         max_sales = customer_sales.max()
#         max_similar = similar_counts.max() if len(similar_counts) > 0 else 1

#         results = []
#         for c in customer_counts.index:
#             score = (
#                 (customer_counts[c] / max_count) * 0.5 +
#                 (customer_sales[c] / max_sales) * 0.3 +
#                 (similar_counts.get(c, 0) / max_similar) * 0.2
#             )
#             results.append([c, customer_counts[c], customer_sales[c], round(score, 3), int(score * 100)])

#         for i in self.tree.get_children():
#             self.tree.delete(i)

#         avg_score = 0
#         if len(results) > 0:
#             for r in results:
#                 self.tree.insert("", "end", values=r)
#             avg_score = sum([r[3] for r in results]) / len(results)

#         expected_value = avg_score * df_p["QUANTITYORDERED"].mean() * df_p["PRICEEACH"].mean()
#         production_cost = df_p["PRICEEACH"].mean() * 0.6

#         worth = (expected_value / production_cost) * 100 if production_cost > 0 else 0

#         self.result_lbl.config(text=f"ارزش تولید مجدد این محصول ≈ {worth:.1f}%")



# import tkinter as tk
# from tkinter import ttk, messagebox
# import pandas as pd
# import numpy as np
# from pathlib import Path
# from tensorflow.keras.models import Sequential, load_model
# from tensorflow.keras.layers import Dense, Dropout
# from tensorflow.keras.optimizers import Adam
# from sklearn.preprocessing import LabelEncoder, StandardScaler
# from sklearn.model_selection import train_test_split
# import matplotlib.pyplot as plt
# from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# # مسیر دیتاست Churn (نمونه)
# churn_dataset_path = Path(r"D:\deeplearning\project\CRM2\churn_data_sample.csv")

# if not churn_dataset_path.exists():
#     raise FileNotFoundError(f"Churn dataset not found at: {churn_dataset_path}")

# churn_df = pd.read_csv(churn_dataset_path)

# # پیش‌پردازش ساده برای مثال
# def preprocess_churn(df):
#     df = df.copy()
#     # تبدیل ستون‌های عددی و دسته‌ای به فرم مناسب
#     le = LabelEncoder()
#     for col in df.select_dtypes(include=['object']).columns:
#         df[col] = le.fit_transform(df[col])
#     # پر کردن مقادیر خالی
#     df.fillna(0, inplace=True)
#     X = df.drop(columns=['Churn'])
#     y = df['Churn'].values
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X)
#     return X_scaled, y, scaler

# X, y, scaler = preprocess_churn(churn_df)

# # ساخت مدل MLP ساده
# def build_churn_model(input_dim):
#     model = Sequential([
#         Dense(64, input_dim=input_dim, activation='relu'),
#         Dropout(0.3),
#         Dense(32, activation='relu'),
#         Dropout(0.2),
#         Dense(1, activation='sigmoid')
#     ])
#     model.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])
#     return model

# churn_model_path = Path(r"D:\deeplearning\project\CRM2\models\churn_model.h5")
# if churn_model_path.exists():
#     churn_model = load_model(churn_model_path)
# else:
#     churn_model = build_churn_model(X.shape[1])
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#     churn_model.fit(X_train, y_train, epochs=20, batch_size=16, validation_data=(X_test, y_test), verbose=0)
#     churn_model.save(churn_model_path)


# class ChurnFrame(tk.Frame):
#     def __init__(self, master, df, model, scaler, **kwargs):
#         super().__init__(master, **kwargs)
#         self.df = df.copy()
#         self.model = model
#         self.scaler = scaler

#         tk.Label(self, text="پیش‌بینی مشتری از دست رفته (Churn)", font=("Arial", 22, "bold")).pack(pady=12)

#         # Treeview برای نمایش نتایج
#         columns = ("customer", "probability")
#         self.tree = ttk.Treeview(self, columns=columns, show="headings", height=15)
#         self.tree.pack(fill="both", expand=True, padx=12, pady=12)
#         self.tree.heading("customer", text="Customer")
#         self.tree.heading("probability", text="Churn Probability (%)")
#         self.tree.column("customer", width=250)
#         self.tree.column("probability", width=150, anchor="center")

#         self.predict_btn = tk.Button(
#             self,
#             text="پیش‌بینی Churn",
#             bg="#e74c3c",
#             fg="white",
#             font=("Arial", 14, "bold"),
#             relief="flat",
#             command=self.run_prediction
#         )
#         self.predict_btn.pack(pady=10, ipadx=20, ipady=6)

#     def run_prediction(self):
#         # پیش‌پردازش و پیش‌بینی
#         X_scaled, _, _ = preprocess_churn(self.df)
#         probs = self.model.predict(X_scaled, verbose=0).flatten()

#         # پاک کردن Treeview
#         for r in self.tree.get_children():
#             self.tree.delete(r)

#         # نمایش در Treeview
#         for idx, cust in enumerate(self.df['CUSTOMERNAME']):
#             self.tree.insert("", "end", values=(cust, f"{probs[idx]*100:.2f}%"))

#         # نمودار
#         fig, ax = plt.subplots(figsize=(6,4))
#         ax.hist(probs*100, bins=20, color="#e74c3c", alpha=0.7)
#         ax.set_xlabel("Churn Probability (%)")
#         ax.set_ylabel("Number of Customers")
#         ax.set_title("Distribution of Churn Probability")
#         for widget in getattr(self, "canvas_frame", []):
#             widget.destroy()
#         self.canvas_frame = tk.Frame(self)
#         self.canvas_frame.pack(pady=10, fill="both", expand=True)
#         canvas = FigureCanvasTkAgg(fig, master=self.canvas_frame)
#         canvas.draw()
#         canvas.get_tk_widget().pack(fill="both", expand=True)


    


# class CRMAppWithChurn(CRMApp):
#     def show_content(self, name):
#         self.clear_main()
#         if name == "HOME":
#             self.show_home()
#             return
#         elif name == "پیش‌بینی مشتری از دست رفته (Churn)":
#             widget = ChurnFrame(self.main_frame, churn_df, churn_model, scaler)
#         else:
#             super().show_content(name)
#             return
#         widget.pack(fill="both", expand=True)

# if __name__ == "__main__":
#     root = tk.Tk()
#     app = CRMAppWithChurn(root)
#     # اضافه کردن دکمه جدید به منو
#     churn_btn = tk.Button(app.sidebar, text="پیش‌بینی مشتری از دست رفته (Churn)",
#                           font=("Arial", 12), bg="#34495e", fg="white",
#                           relief="flat", command=lambda: app.show_content("پیش‌بینی مشتری از دست رفته (Churn)"))
#     churn_btn.pack(fill="x", pady=5, padx=12)
#     root.mainloop()




#     def build_churn_model(input_dim):
#         model = Sequential([
#             Dense(64, input_dim=input_dim, activation='relu'),
#             Dropout(0.3),
#             Dense(32, activation='relu'),
#             Dropout(0.2),
#             Dense(1, activation='sigmoid')
#         ])
#         model.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])
#         return model

#     churn_model_path = Path(r"D:\deeplearning\project\CRM2\models\churn_model.h5")
#     if churn_model_path.exists():
#         churn_model = load_model(churn_model_path)
#     else:
#         churn_model = build_churn_model(X.shape[1])
#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#         churn_model.fit(X_train, y_train, epochs=20, batch_size=16, validation_data=(X_test, y_test), verbose=0)
#         churn_model.save(churn_model_path)











# class AnalyticsFrame(tk.Frame):
#     def __init__(self, parent, df):
#         super().__init__(parent)
        
#         self.df = df.copy()
        
#         canvas = tk.Canvas(self)
#         scrollbar = tk.Scrollbar(self, orient="vertical", command=canvas.yview)
#         scroll_frame = tk.Frame(canvas)

#         scroll_frame.bind(
#             "<Configure>",
#             lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
#         )

#         canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
#         canvas.configure(yscrollcommand=scrollbar.set)

#         canvas.pack(side="left", fill="both", expand=True)
#         scrollbar.pack(side="right", fill="y")

#         self.images = []

#         self.build_charts(scroll_frame)
    

#     def build_charts(self, container):

#         import matplotlib.pyplot as plt
#         import seaborn as sns
#         from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        
#         sns.set(style="whitegrid")

#         def add_chart(title, fig):
#             lbl = tk.Label(container, text=title, font=("Arial", 16, "bold"))
#             lbl.pack(pady=10)

#             canvas = FigureCanvasTkAgg(fig, master=container)
#             canvas.get_tk_widget().pack(pady=10)

#         # 1) Top 10 Products by Sales
#         fig1 = plt.figure(figsize=(8, 4))
#         top_products = (
#             self.df.groupby("PRODUCTLINE")["SALES"]
#             .sum()
#             .sort_values(ascending=False)
#             .head(10)
#         )
#         sns.barplot(x=top_products.values, y=top_products.index)
#         plt.title("Top 10 Products by Sales")
#         add_chart("پرفروش‌ترین محصولات", fig1)

#         fig2 = plt.figure(figsize=(8, 4))
#         trend = (
#             self.df.groupby("MONTH_ID")["SALES"]
#             .sum()
#             .sort_index()
#         )
#         sns.lineplot(x=trend.index, y=trend.values, marker="o")
#         plt.title("Monthly Sales Trend")
#         add_chart("روند فروش ماهیانه", fig2)

#         fig3 = plt.figure(figsize=(6, 6))
#         share = self.df.groupby("PRODUCTLINE")["SALES"].sum()
#         plt.pie(share, labels=share.index, autopct="%1.1f%%")
#         plt.title("Market Share")
#         add_chart("سهم هر محصول از بازار", fig3)

#         fig4 = plt.figure(figsize=(8, 4))
#         freq = self.df["CUSTOMERNAME"].value_counts()
#         sns.histplot(freq, bins=20)
#         plt.title("Customer Purchase Frequency")
#         add_chart("تکرار خرید مشتریان", fig4)

#         fig5 = plt.figure(figsize=(8, 4))
#         country_sales = (
#             self.df.groupby("COUNTRY")["SALES"]
#             .sum()
#             .sort_values(ascending=False)
#             .head(15)
#         )
#         sns.barplot(x=country_sales.values, y=country_sales.index)
#         plt.title("Sales by Country")
#         add_chart("فروش بر اساس کشور", fig5)


#         fig6 = plt.figure(figsize=(8, 6))
#         num_df = self.df.select_dtypes(include=['int', 'float'])
#         sns.heatmap(num_df.corr(), annot=True, cmap="coolwarm")
#         plt.title("Correlation Heatmap")
#         add_chart("هیت‌مپ همبستگی", fig6)


#         fig7 = plt.figure(figsize=(8, 4))
#         profit = (
#             self.df.groupby("PRODUCTLINE")["SALES"]
#             .mean()
#             .sort_values(ascending=False)
#         )
#         sns.barplot(x=profit.values, y=profit.index)
#         plt.title("Average Profit per Product")
#         add_chart("میانگین سود هر محصول", fig7)













# class CRMApp:
#     def __init__(self, master):
#         self.master = master
#         master.title("CRM برای شما")
#         master.geometry("1200x780")

#         # Sidebar
#         self.sidebar = tk.Frame(master, width=260, bg="#2c3e50")
#         self.sidebar.pack(side="left", fill="y")
#         tk.Label(self.sidebar, text="ابزارها:", bg="#2c3e50", fg="white", font=("Arial", 14, "bold")).pack(pady=20)

#         menu_items = [
#             "HOME",
#             "پیش‌بینی آینده فروش محصول",
#             "بیشترین فروش محصول",
#             "سفارشات مشتریان",
#             "تحلیل کلی مشتریان",
#             "اطلاعات مشتریان",
#             "هر محصول برای کدام مشتری؟",
#             "پیشبینی مشتری ای که ممکن است از دست برود"
#             "نمودارهای تحلیلی"
#         ]

#         for item in menu_items:
#             btn = tk.Button(self.sidebar, text=item, font=("Arial", 12), bg="#34495e", fg="white",
#                             relief="flat", command=lambda n=item: self.show_content(n))
#             btn.pack(fill="x", pady=5, padx=12)


#         self.main_frame = tk.Frame(master, bg="#ecf0f1")
#         self.main_frame.pack(side="left", fill="both", expand=True)

#         self.current_widget = None
#         self.show_home()

#     def clear_main(self):
#         for w in self.main_frame.winfo_children():
#             w.destroy()

#     def show_home(self):
#         self.clear_main()
#         tk.Label(self.main_frame, text="CRM برای شما", font=("Arial", 28, "bold")).pack(pady=40)
#         tk.Label(self.main_frame, text="از منوی سمت چپ یک بخش را انتخاب کنید.", font=("Arial", 14)).pack(pady=10)

#     def show_content(self, name):
#         self.clear_main()
#         if name == "HOME":
#             self.show_home()
#             return
#         elif name == "پیش‌بینی آینده فروش محصول":
#             widget = FutureSalesFrame(self.main_frame, df)
#         elif name == "بیشترین فروش محصول":
#             widget = TopSalesFrame(self.main_frame, df)
#         elif name == "سفارشات مشتریان":
#             widget = CustomerOrdersFrame(self.main_frame, df)
#         elif name == "تحلیل کلی مشتریان":
#             widget = CustomersFrame(self.main_frame, df)
#         elif name == "اطلاعات مشتریان":  # <-- اضافه شد
#             widget = CustomerInfoFrame(self.main_frame, df)
#         elif name == "هر محصول برای کدام مشتری؟":
#             widget = ProductCustomerFrame(self.main_frame, df),
#         elif name == "پیشبینی مشتری ای که ممکن است از دست برود":
#             widget = AnalyticsFrame(self.main_frame, df)
#         elif name == "نمودارهای تحلیلی":
#             widget = AnalyticsFrame(self.main_frame, df)
    

#         else:
#             widget = tk.Label(self.main_frame, text=f"این قسمت مربوط به {name} است.", font=("Arial", 16))
#         widget.pack(fill="both", expand=True)


# if __name__ == "__main__":
#     root = tk.Tk()
#     app = CRMApp(root)
#     root.mainloop()





