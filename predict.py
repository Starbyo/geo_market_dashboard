
import joblib
import pandas as pd

model = joblib.load("model.pkl")

today = pd.DataFrame([{
"oil_return": 0.02,
"gold_return": 0.01
}])

prediction = model.predict(today)

if prediction[0] == 1:
    print("Risk ON: Stocks likely up")
else:
    print("Risk OFF: Stocks likely down")
