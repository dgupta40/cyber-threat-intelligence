import pandas as pd, joblib, pathlib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

df = pd.read_parquet("data/processed/tfidf.parquet").dropna()
X_train, X_test, y_train, y_test = train_test_split(
    df.drop(columns=["label"]), df["label"], test_size=0.2, random_state=42)

clf = LogisticRegression(max_iter=1000).fit(X_train, y_train)
print(classification_report(y_test, clf.predict(X_test)))
pathlib.Path("models").mkdir(exist_ok=True)
joblib.dump(clf, "models/logreg.pkl")
