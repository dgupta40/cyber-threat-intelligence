import re, json, glob, pathlib, pandas as pd, spacy
from sklearn.feature_extraction.text import TfidfVectorizer

nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
RAW_GLOB = "data/raw/*.json"
OUT_FILE = "data/processed/tfidf.parquet"

def normalise(txt: str) -> str:
    txt = re.sub(r"http\S+|www\.\S+", " ", txt)
    txt = re.sub(r"<[^>]+>", " ", txt)
    txt = re.sub(r"[^A-Za-z0-9\s]+", " ", txt)
    doc = nlp(txt.lower())
    return " ".join(t.lemma_ for t in doc if not t.is_stop)

def build():
    corpus, labels, meta = [], [], []
    for fp in glob.glob(RAW_GLOB):
        d = json.load(open(fp, encoding="utf-8"))

        # Use all English descriptions concatenated + WP-Scan PoC if any
        descs = [dd["value"] for dd in d["cve"]["descriptions"] if dd["lang"] == "en"]
        if wp := d.get("wp_scan", {}):
            if poc := wp.get("poc"):
                descs.append(poc)

        corpus.append(normalise(" ".join(descs)))
        labels.append(wp.get("type") or "unknown")
        meta.append({"source": d["source"], "id": d["cve"]["id"]})

    tfidf = TfidfVectorizer(max_features=8000).fit_transform(corpus)
    df = pd.DataFrame(tfidf.toarray())
    df["label"] = labels
    df["source"] = [m["source"] for m in meta]
    df["cve_id"] = [m["id"] for m in meta]
    pathlib.Path(OUT_FILE).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_FILE)
    print(f"Saved vector matrix → {OUT_FILE}")

if __name__ == "__main__":
    build()
