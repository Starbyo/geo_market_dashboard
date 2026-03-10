import joblib
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier

def load_or_train_model():
    if os.path.exists("model.pkl"):
        return joblib.load("model.pkl")

    np.random.seed(42)
    n = 1000
    oil_return  = np.concatenate([np.random.normal( 0.01, 0.02, n//2),
                                   np.random.normal(-0.01, 0.02, n//2)])
    gold_return = np.concatenate([np.random.normal(-0.005, 0.015, n//2),
                                   np.random.normal( 0.010, 0.015, n//2)])
    target      = np.array([1]*(n//2) + [0]*(n//2))

    df = pd.DataFrame({"oil_return": oil_return,
                        "gold_return": gold_return,
                        "target": target})
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(df[["oil_return", "gold_return"]], df["target"])
    joblib.dump(model, "model.pkl")
    print("Model trained and saved as model.pkl")
    return model

model = load_or_train_model()

today = pd.DataFrame([{
    "oil_return": 0.02,
    "gold_return": 0.01
}])

prediction = model.predict(today)[0]
proba      = model.predict_proba(today)[0]
confidence = int(max(proba) * 100)

if prediction == 1:
    print(f"Risk ON: Stocks likely up ({confidence}% confidence)")
else:
    print(f"Risk OFF: Stocks likely down ({confidence}% confidence)")
