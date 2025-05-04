import pandas as pd, json, glob, pathlib
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

sid  = SentimentIntensityAnalyzer()
RAW  = "data/raw/*.json"
OUT  = "data/processed/risk_scores.csv"

def cvss_to_risk(score: float | None):
    if score is None:
        return None
    if score >= 7.0: return "High"
    if score >= 4.0: return "Medium"
    return "Low"

rows = []
for fp in glob.glob(RAW):
    d = json.load(open(fp, encoding="utf-8"))
    cid = d["cve"]["id"]

    # prefer CVSS if present
    cvss = (d.get("wp_scan", {}) or {}).get("cvss")
    risk = cvss_to_risk(float(cvss) if cvss else None)

    if risk is None:
        # fallback to sentiment of description
        text = " ".join(dd["value"] for dd in d["cve"]["descriptions"] if dd["lang"] == "en")
        score = sid.polarity_scores(text)["compound"]
        risk = "High" if score < -0.4 else "Medium" if score < 0 else "Low"

    rows.append({"cve_id": cid, "risk": risk, "cvss": cvss})

pd.DataFrame(rows).to_csv(OUT, index=False)
print(f"Risk scores saved → {OUT}")
