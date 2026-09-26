# train_churn_model.py
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

# ------------------ 1. بارگذاری دیتاست ------------------
data_path = Path(r"D:\project\CRM2\sales_data_sample.csv")
if not data_path.exists():
    raise FileNotFoundError(f"Dataset not found at: {data_path}")

df = pd.read_csv(data_path, encoding='cp1252')

# فرض کنید ستون churn با نام 'Churn' داریم (0: active, 1: churned)
# اگر هنوز ندارید، می‌توانید بر اساس DEALSIZE یا STATUS یک ستون بسازید
if 'Churn' not in df.columns:
    df['Churn'] = df['DEALSIZE'].apply(lambda x: 1 if x=='Small' else 0)  # نمونه فرضی

# ------------------ 2. ویژگی‌ها و پیش‌پردازش ------------------
features = ['QUANTITYORDERED','PRICEEACH','ORDERLINENUMBER','PRODUCTLINE',
            'MSRP','PRODUCTCODE','COUNTRY','TERRITORY','MONTH_ID','YEAR_ID']
target = 'Churn'

X = df[features]
y = df[target]

numeric_features = X.select_dtypes(include=['int64','float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

numeric_transformer = Pipeline([
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline([
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

X_processed = preprocessor.fit_transform(X)

# ------------------ 3. تقسیم داده ------------------
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

# ------------------ 4. آموزش مدل ------------------
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# ------------------ 5. ارزیابی ------------------
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# ------------------ 6. ذخیره مدل و preprocessor ------------------
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/churn_model.pkl")
joblib.dump(preprocessor, "models/preprocessor.pkl")

print("Churn model and preprocessor saved successfully in 'models/' folder!")
