#!/usr/bin/env python3
"""
raw_eda_full.py
───────────────
Comprehensive Raw‑Data EDA for NVD + TheHackerNews, updated:

  • Correct THN date indexing (falls back to ingest_ts_thn)
  • NVD loader extracts CVE IDs for lag calculation
  • All monthly resamples use "ME" to silence FutureWarning
  • Fixed lag histogram logic with .dt.days
  • Guard against empty text for word‑cloud

Usage:
  pip install pandas matplotlib seaborn squarify wordcloud
  python raw_eda_full.py \
    --nvd data/raw/nvd/nvd.json \
    --thn data/raw/hackernews/hackernews.json \
    --out charts/full
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import squarify
from wordcloud import WordCloud

plt.rcParams.update({"figure.autolayout": True})


def load_nvd_full(path: Path) -> pd.DataFrame:
    recs = []
    with open(path, encoding="utf-8") as f:
        root = json.load(f)
    items = root.get("vulnerabilities") or root.get("CVE_Items")
    if items is None:
        raise ValueError("Unrecognized NVD schema")
    for item in items:
        cve_block = item.get("cve") or item
        cve_id    = cve_block.get("id") or cve_block.get("CVE_data_meta", {}).get("ID", "UNKNOWN")
        pub       = cve_block.get("published") or cve_block.get("publishedDate")
        mets = cve_block.get("metrics", {})
        score, sev = None, "UNKNOWN"
        for key in ("cvssMetricV31","cvssMetricV30"):
            if key in mets:
                data = mets[key][0]["cvssData"]
                score = float(data.get("baseScore", 0.0))
                sev   = data.get("baseSeverity", "UNKNOWN")
                break
        cwe = "UNKNOWN"
        for w in cve_block.get("weaknesses", []):
            for d in w.get("description", []):
                cwe = d.get("value", "UNKNOWN")
                break
            if cwe != "UNKNOWN": break
        vp = "UNKNOWN"
        for cfg in cve_block.get("configurations", []):
            for node in cfg.get("nodes", []):
                for m in node.get("cpeMatch", []):
                    parts = m.get("criteria", "").split(":")
                    if len(parts) >= 5:
                        vp = f"{parts[3]}:{parts[4]}"
                        break
                if vp != "UNKNOWN": break
            if vp != "UNKNOWN": break
        if pub:
            recs.append((pd.to_datetime(pub), cve_id,
                         score if score is not None else float("nan"),
                         sev, cwe, vp))
    df = pd.DataFrame(
        recs,
        columns=["published","id","cvssScore","severity","cwe","vendor_product"]
    ).set_index("published")
    return df


def load_thn(path: Path) -> pd.DataFrame:
    rows = json.load(open(path, encoding="utf-8"))
    df = pd.json_normalize(rows)
    df["date_parsed"]   = pd.to_datetime(df["date"], errors="coerce")
    df["ingest_parsed"] = pd.to_datetime(df["ingest_ts_thn"], errors="coerce")
    df["use_date"]      = df["date_parsed"].fillna(df["ingest_parsed"])
    return df.set_index("use_date").sort_index()


def timeline_or_bar(df: pd.DataFrame, rule: str, title: str, out: Path):
    uniq = df.index.normalize().nunique()
    if uniq < 3:
        print(f"⚠️ Only {uniq} unique dates for '{title}' → bar fallback")
        df.index.normalize().value_counts()\
           .sort_index()\
           .plot(kind="bar", figsize=(8,4), title=title)
        plt.xticks(rotation=45, ha="right")
    else:
        df.resample(rule).size()\
           .plot(figsize=(10,4), title=title)
    plt.ylabel("Count")
    plt.savefig(out)
    plt.close()


def plot_severity_mix(df: pd.DataFrame, out: Path):
    by_year = df.reset_index()\
                .groupby([df.index.year, "severity"])\
                .size()\
                .unstack(fill_value=0)
    if by_year.shape[0] < 2:
        print("⚠️ Not enough years for severity mix; skipping")
        return
    by_year.plot(kind="bar", stacked=True, figsize=(12,5),
                 title="CVSS‑v3 Severity mix by year")
    plt.ylabel("Count")
    plt.savefig(out); plt.close()


def plot_top_cwe(df: pd.DataFrame, out: Path, n=15):
    counts = df["cwe"].value_counts().head(n)
    if counts.empty:
        print("⚠️ No CWE data; skipping")
        return
    counts.sort_values().plot(kind="barh", figsize=(8,6),
                              title=f"Top‑{n} CWE categories")
    plt.xlabel("CVE count")
    plt.savefig(out); plt.close()


def plot_vendor_treemap(df: pd.DataFrame, out: Path):
    crit = df[df["severity"] == "CRITICAL"]
    top = crit["vendor_product"].value_counts().head(30)
    if top.empty:
        print("⚠️ No critical CVEs; skipping treemap")
        return
    squarify.plot(sizes=top.values, label=top.index, alpha=0.8, pad=True)
    plt.axis("off"); plt.title("Critical CVEs by vendor:product")
    plt.gcf().set_size_inches(11,7)
    plt.savefig(out); plt.close()


def plot_thn_tags(df: pd.DataFrame, out: Path, n=20):
    tags = df["tags"].explode().dropna()
    counts = tags.value_counts().head(n)
    if counts.empty:
        print("⚠️ No THN tags; skipping")
        return
    counts.sort_values().plot(kind="barh", figsize=(8,6),
                              title=f"Top‑{n} THN tags")
    plt.xlabel("Article count")
    plt.savefig(out); plt.close()


def plot_cves_per_article(df: pd.DataFrame, out: Path):
    counts = df["cves"].apply(lambda x: len(x) if isinstance(x, list) else 0)
    if counts.nunique() < 2:
        print("⚠️ No CVE‑count variation; skipping")
        return
    counts.plot(kind="hist", bins=20, figsize=(10,4),
                title="Number of CVEs mentioned per article")
    plt.xlabel("CVE count"); plt.ylabel("Articles")
    plt.savefig(out); plt.close()


def plot_dual_timeline(df_nvd: pd.DataFrame, df_thn: pd.DataFrame, out: Path):
    cvm = df_nvd.resample("ME").size()
    thw = df_thn.resample("W").size()
    fig, ax1 = plt.subplots(figsize=(12,5))
    ax1.plot(cvm.index, cvm, color="tab:blue"); ax1.set_ylabel("CVEs/month", color="tab:blue")
    ax2 = ax1.twinx()
    ax2.plot(thw.index, thw, color="tab:orange", alpha=0.6); ax2.set_ylabel("THN articles/week", color="tab:orange")
    plt.title("CVE Volume vs THN Coverage")
    fig.tight_layout(); fig.savefig(out); plt.close()


def plot_lag_hist(df_thn: pd.DataFrame, df_nvd: pd.DataFrame, out: Path):
    expl = df_thn.explode("cves").dropna(subset=["cves"])
    first = expl.groupby("cves") \
                .apply(lambda grp: grp.index.min()) \
                .rename("first_thn")

    nvd_idx = df_nvd.reset_index()[["id","published"]].set_index("id")
    merged  = first.to_frame().join(nvd_idx, how="inner")
    merged["lag"] = (merged["first_thn"] - merged["published"]).dt.days
    merged["lag"].plot(kind="hist", bins=50, figsize=(10,4),
                       title="Lag: NVD publish → first THN mention")
    plt.xlabel("Days"); plt.savefig(out); plt.close()


def plot_avg_cvss(df_nvd: pd.DataFrame, out: Path):
    avg = df_nvd["cvssScore"].resample("ME").mean()
    if avg.isna().all():
        print("⚠️ No CVSS data; skipping avg‑score plot")
        return
    avg.plot(figsize=(12,4), title="Monthly average CVSS‑v3 score")
    plt.ylabel("Avg CVSS"); plt.savefig(out); plt.close()


def plot_wordcloud(path: Path, out: Path):
    texts = []
    root  = json.load(open(path, encoding="utf-8"))
    items = root.get("CVE_Items") or root.get("vulnerabilities")
    for item in items:
        c = item.get("cve") or item
        for d in c.get("description", {}).get("description_data", []):
            val = d.get("value", "").strip()
            if val:
                texts.append(val)
    if not texts:
        print("⚠️ No text found for word‑cloud; skipping")
        return
    combined = " ".join(texts)
    wc = WordCloud(width=800, height=400, max_words=200).generate(combined)
    plt.figure(figsize=(10,5)); plt.imshow(wc, interpolation="bilinear")
    plt.axis("off"); plt.title("Raw NVD descriptions word‑cloud")
    plt.savefig(out); plt.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--nvd", required=True, type=Path)
    p.add_argument("--thn", required=True, type=Path)
    p.add_argument("--out", default="charts/full", type=Path)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    print("[*] Loading full NVD…")
    df_nvd = load_nvd_full(args.nvd)
    print(f"    rows={len(df_nvd):,}, dates={df_nvd.index.normalize().nunique()}")

    print("[*] Loading THN…")
    df_thn = load_thn(args.thn)
    print(f"    rows={len(df_thn):,}, dates={df_thn.index.normalize().nunique()}")

    timeline_or_bar(df_nvd, "ME", "CVEs published per month", args.out/"nvd_cves_per_month.png")
    timeline_or_bar(df_thn, "W", "THN articles per week", args.out/"thn_articles_per_week.png")

    plot_severity_mix(df_nvd, args.out/"severity_mix_by_year.png")
    plot_top_cwe(df_nvd, args.out/"top_cwe.png")
    plot_vendor_treemap(df_nvd, args.out/"vendor_treemap.png")
    plot_thn_tags(df_thn, args.out/"thn_tags.png")
    plot_cves_per_article(df_thn, args.out/"cves_per_article_hist.png")

    plot_dual_timeline(df_nvd, df_thn, args.out/"dual_timeline.png")
    plot_lag_hist(df_thn, df_nvd, args.out/"lag_hist.png")
    plot_avg_cvss(df_nvd, args.out/"avg_cvss.png")
    plot_wordcloud(args.nvd, args.out/"nvd_wordcloud.png")

    print("✓ All charts saved to", args.out)


if __name__ == "__main__":
    main()
