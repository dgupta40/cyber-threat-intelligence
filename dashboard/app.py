import streamlit as st, pandas as pd, plotly.express as px, pathlib, datetime

st.set_page_config(page_title="Threat-Intel Dashboard", layout="wide")

DATA_DIR = pathlib.Path("data/processed")
tfidf = pd.read_parquet(DATA_DIR / "tfidf.parquet")
risk  = pd.read_csv(DATA_DIR / "risk_scores.csv")
anoms = pd.read_csv(DATA_DIR / "anomalies.csv")

st.sidebar.title("Filters")
src = st.sidebar.multiselect("Source", tfidf["source"].unique(),
                             default=list(tfidf["source"].unique()))

filtered = tfidf[tfidf["source"].isin(src)]

# --- high level KPIs
col1, col2 = st.columns(2)
col1.metric("Total Threats", len(filtered))
col2.metric("Unique CVEs", filtered["cve_id"].nunique())

# --- charts
st.plotly_chart(
    px.bar(filtered["label"].value_counts().reset_index(),
           x="index", y="label", labels={"index":"Category","label":"Count"},
           title="Threat categories")
)

st.plotly_chart(
    px.histogram(risk, x="risk", title="Risk Levels")
)

# --- data table with extras --------------------------------------------------
st.subheader("🗒️ Detailed Threats Table")

# join risk + tfidf meta to display extra fields
tbl = filtered[["cve_id","label"]].merge(risk, on="cve_id", how="left")

# Add age column if possible (published date is inside raw; quick lambda fetch)
def age_days(cve_id):
    raw_file = next(pathlib.Path("data/raw").glob(f"*{cve_id}.json"), None)
    if not raw_file: return None
    import json, dateutil.parser
    pub = dateutil.parser.isoparse(json.load(open(raw_file))["cve"]["published"])
    return (datetime.datetime.utcnow() - pub).days
tbl["age_days"] = tbl["cve_id"].apply(age_days)

st.dataframe(tbl.sort_values("age_days", ascending=True).reset_index(drop=True))
