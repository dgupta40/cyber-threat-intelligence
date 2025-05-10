#!/usr/bin/env python3
"""
merge_master.py
────────────────
Join THN articles to NVD metadata, producing one row per (article, CVE).

Steps:
  1) Load cleaned master parquet (raw rows with source tag)
  2) Split into THN vs NVD DataFrames
  3) Extract `cve_id` and parse dates
  4) Explode THN on its `cves` list → one entry per article–CVE
  5) Left‑merge THN → NVD on `cve_id` to bring in vulnerability fields
  6) Compute `publish_gap_hours` and a simple `sent_bin`
  7) Write out `master.parquet` for modeling/dashboarding

Usage:
  python merge_master.py --proc data/processed/master_YYYYMMDD_HHMM.parquet --out data/processed/master.parquet
"""
import argparse
from pathlib import Path

import pandas as pd

def main(proc_path: Path, out_path: Path):
    # 1) Load the cleaned, source‑tagged table
    df = pd.read_parquet(proc_path)

    # 2) Split by source
    df_thn = df[df['source']=='thehackernews'].copy()
    df_nvd = df[df['source']=='nvd'].copy()

    # 3a) Normalize NVD CVE ID & parse publish date
    df_nvd['cve_id'] = df_nvd['cve'].apply(lambda c: c.get('id') if isinstance(c, dict) else None)
    df_nvd['published'] = pd.to_datetime(df_nvd['published'], utc=True, errors='coerce')

    # 3b) Explode THN on its list of CVEs, parse article date
    df_thn = df_thn.rename(columns={'cves':'cve_id_list'})
    df_thn = df_thn.explode('cve_id_list').dropna(subset=['cve_id_list'])
    df_thn['cve_id'] = df_thn['cve_id_list']
    df_thn['article_date'] = pd.to_datetime(df_thn['date'], utc=True, errors='coerce')

    # 4) Left‑merge THN → NVD on cve_id
    keep = ['cve_id','cvssScore','severity_bin','cwe','vendor_product','published']
    master = df_thn.merge(df_nvd[keep], on='cve_id', how='left', suffixes=('_thn','_nvd'))

    # 5a) Feature: gap in hours between article and CVE publish
    master['publish_gap_hours'] = (
        master['article_date'] - master['published']
    ).dt.total_seconds().div(3600)

    # 5b) Feature: simple sentiment bin
    master['sent_bin'] = pd.cut(
        master['sentiment'],
        bins=[-1.0, -0.1, 0.1, 1.0],
        labels=['neg','neu','pos']
    )

    # 6) Persist final master
    out_path.parent.mkdir(parents=True, exist_ok=True)
    master.to_parquet(out_path, index=False, compression='zstd')
    print(f"✓ Wrote merged master to {out_path}")

if __name__=='__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--proc', required=True, help="Cleaned master parquet")
    p.add_argument('--out',   default='data/processed/master.parquet',
                   help="Output path for merged master")
    args = p.parse_args()
    main(Path(args.proc), Path(args.out))