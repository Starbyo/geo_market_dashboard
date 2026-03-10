
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

data = pd.read_csv("data/markets.csv")

data["target"] = (data["sp500_return"] > 0).astype(int)

features = data[["oil_return","gold_return"]]
target = data["target"]

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(features, target)

joblib.dump(model, "model.pkl")

print("Model trained and saved as model.pkl")
