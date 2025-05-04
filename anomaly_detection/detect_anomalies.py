import pandas as pd, joblib, numpy as np, pathlib
from sklearn.ensemble import IsolationForest

X = pd.read_parquet("data/processed/tfidf.parquet").drop(columns=["label"])
iso = IsolationForest(contamination=0.02, random_state=42).fit(X)
scores = iso.decision_function(X)
pd.Series(scores, name="anomaly_score").to_csv("data/processed/anomalies.csv", index=False)
joblib.dump(iso, "models/iso.pkl")
