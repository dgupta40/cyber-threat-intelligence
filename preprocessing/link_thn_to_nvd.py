#!/usr/bin/env python3
"""
link_thn_to_nvd.py — OPTIMIZED VERSION
 Uses a fast in-memory CVE → article index mapping
 Handles NumPy arrays properly
 Stores variable-length lists correctly
"""

import pandas as pd
import numpy as np
from pathlib import Path

def link_thn_to_nvd(master_path="data/processed/master.parquet",
                    parquet_output="data/processed/master_linked.parquet",
                    csv_output="data/processed/master_linked.csv"):
    print(f"Loading master dataset from: {master_path}")
    df = pd.read_parquet(master_path)
    df_nvd = df[df["source"] == "nvd"].copy()
    df_thn = df[df["source"] == "thehackernews"].copy()
    
    print(f"Total records: {len(df)} (NVD: {len(df_nvd)}, THN: {len(df_thn)})")

    # Build fast CVE → article index mapping
    print("Building CVE → article index mapping...")
    cve_to_articles = {}
    for idx, row in df_thn.iterrows():
        cves = row.get("all_cves", [])
        if isinstance(cves, np.ndarray):
            for cve in cves:
                if cve:  # Skip empty values
                    cve_str = str(cve).strip().upper()
                    cve_to_articles.setdefault(cve_str, []).append(idx)
        elif isinstance(cves, list):
            for cve in cves:
                if cve:  # Skip empty values
                    cve_str = str(cve).strip().upper()
                    cve_to_articles.setdefault(cve_str, []).append(idx)
    
    print(f"Created mapping with {len(cve_to_articles)} unique CVEs")
    
    # Process NVD entries in batches for better performance
    batch_size = 5000
    total_batches = (len(df_nvd) + batch_size - 1) // batch_size
    
    # Create empty columns in df_nvd first
    df_nvd["n_articles"] = 0
    df_nvd["linked_articles"] = None  # Will store lists
    df_nvd["earliest_article_date"] = pd.NaT
    
    linked_count = 0
    print(f"Processing {len(df_nvd)} NVD entries in {total_batches} batches...")
    
    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(df_nvd))
        
        batch_df = df_nvd.iloc[start_idx:end_idx]
        
        for i, (idx, row) in enumerate(batch_df.iterrows()):
            cve = str(row.get("primary_cve", "")).strip().upper()
            article_ids = cve_to_articles.get(cve, [])
            
            if article_ids:
                linked_count += 1
                # Store count and list directly in df_nvd
                df_nvd.at[idx, "n_articles"] = len(article_ids)
                df_nvd.at[idx, "linked_articles"] = article_ids.copy()  # Copy to avoid reference issues
                
                # Get earliest date
                try:
                    dates = df_thn.loc[article_ids, "published_dt"].dropna()
                    if not dates.empty:
                        df_nvd.at[idx, "earliest_article_date"] = dates.min()
                except Exception as e:
                    pass  # Keep as NaT
        
        if (batch_idx + 1) % 10 == 0 or batch_idx == total_batches - 1:
            print(f" Processed {end_idx}/{len(df_nvd)} records ({(batch_idx+1)/total_batches*100:.1f}%)")
    
    print(f"Found links for {linked_count}/{len(df_nvd)} NVD records ({linked_count/len(df_nvd)*100:.1f}%)")

    # Update master dataframe one column at a time
    print("Updating master dataframe...")
    df.loc[df_nvd.index, "n_articles"] = df_nvd["n_articles"]
    
    # For linked_articles, we need to handle the lists specially
    for idx, val in df_nvd["linked_articles"].items():
        df.at[idx, "linked_articles"] = val
    
    df.loc[df_nvd.index, "earliest_article_date"] = df_nvd["earliest_article_date"]

    # Verify update worked
    linked_count_final = (df.loc[df['source'] == 'nvd', 'n_articles'] > 0).sum()
    print(f"After update: {linked_count_final} NVD records have links")

    df.to_parquet(parquet_output, index=False)
    df.to_csv(csv_output, index=False)
    print(f" Linking complete. Saved to:")
    print(f" - Parquet: {parquet_output}")
    print(f" - CSV:     {csv_output}")
    
    return True

if __name__ == "__main__":
    link_thn_to_nvd()