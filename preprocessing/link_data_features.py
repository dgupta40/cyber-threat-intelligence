# link_data_features.py
"""
Link THN articles to NVD CVEs based on:
- Mentioned CVEs
- Temporal distance (±3 days)
- Compute similarity (SBERT cosine)
- Output structured features per (CVE, article) pair
Also saves unlinked THN articles for training or analysis
Exports both Parquet and CSV for easy access
"""

def link_thn_to_nvd():
    import pandas as pd
    import numpy as np
    from pathlib import Path
    from datetime import timedelta
    from sklearn.metrics.pairwise import cosine_similarity
    from sentence_transformers import SentenceTransformer

    PROC_DIR = Path("data/processed")
    EMBED_DIR = Path("models")

    # Load latest parquet
    master_file = sorted(PROC_DIR.glob("master_*.parquet"))[-1]
    df = pd.read_parquet(master_file)

    # Split by source
    df_thn = df[df['source'] == 'thehackernews'].copy()
    df_nvd = df[df['source'] == 'nvd'].copy()

    # Load SBERT vectors for THN
    sbert_path = df_thn['sbert_path'].iloc[0]
    emb_thn = np.load(sbert_path)

    # Preprocess NVD: create SBERT vectors
    sbert = SentenceTransformer("all-MiniLM-L6-v2")
    df_nvd['description'] = df_nvd['description'].fillna('')
    emb_nvd = sbert.encode(df_nvd['description'].tolist(), batch_size=64, show_progress_bar=True)

    # Build index for CVEs
    df_nvd.set_index("cve_id", inplace=True)
    cve_to_nvd = df_nvd.to_dict(orient="index")

    # Match THN articles to CVEs
    linked_rows = []
    unlinked_rows = []

    for idx, row in df_thn.iterrows():
        article_dt = row['published_dt']
        article_cves = row['mentioned_cves']

        if not isinstance(article_cves, list) or len(article_cves) == 0:
            unlinked_rows.append(row)
            continue

        article_emb = emb_thn[idx].reshape(1, -1)

        for cve in article_cves:
            if cve in cve_to_nvd:
                nvd_row = cve_to_nvd[cve]
                temporal_dist = abs((article_dt - nvd_row['published_dt']).days)
                nvd_emb = emb_nvd[df_nvd.index.get_loc(cve)].reshape(1, -1)
                sim_score = float(cosine_similarity(article_emb, nvd_emb)[0][0])

                linked_rows.append({
                    "cve_id": cve,
                    "article_id": idx,
                    "temporal_distance": temporal_dist,
                    "text_similarity": sim_score,
                    "cvss_score": nvd_row.get("cvssScore"),
                    "severity_bin": nvd_row.get("severity_bin"),
                    "sentiment": row["sentiment"],
                    "published_article": article_dt,
                    "published_cve": nvd_row["published_dt"],
                    "tokens": row['tokens'],
                    "sbert_path": row['sbert_path']
                })

    # Save linked dataset
    df_linked = pd.DataFrame(linked_rows)
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M")
    out_parquet = PROC_DIR / f"linked_thn_nvd_{timestamp}.parquet"
    out_csv = PROC_DIR / f"linked_thn_nvd_{timestamp}.csv"
    df_linked.to_parquet(out_parquet, index=False)
    df_linked.to_csv(out_csv, index=False)
    print(f"Linked dataset saved to: {out_parquet}\n CSV version saved to: {out_csv}")

    # Save unlinked THN articles
    if unlinked_rows:
        df_unlinked = pd.DataFrame(unlinked_rows)
        unlinked_path = PROC_DIR / f"unlinked_thn_{timestamp}.csv"
        df_unlinked.to_csv(unlinked_path, index=False)
        print(f" Unlinked THN articles saved to: {unlinked_path}")
