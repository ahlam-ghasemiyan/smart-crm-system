
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from pathlib import Path


class CRMModel:
    def __init__(self):
        self.model = RandomForestRegressor(
            n_estimators=200,
            max_depth=None,
            random_state=42
        )

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X_test, y_test):
        preds = self.predict(X_test)
        return {
            "MAE": mean_absolute_error(y_test, preds),
            "MSE": mean_squared_error(y_test, preds),
            "RMSE": mean_squared_error(y_test, preds, squared=False),
            "R2": r2_score(y_test, preds)
        }

    def save(self, folder_path):
        folder = Path(folder_path)
        folder.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, folder / "model.pkl")

    def load(self, folder_path):
        folder = Path(folder_path)
        self.model = joblib.load(folder / "model.pkl")
