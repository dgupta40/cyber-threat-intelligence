#!/usr/bin/env python3
"""
Parquet Viewer - A simplified version that can handle complex data types.
"""

import os
import sys
import pandas as pd
import numpy as np

def view_parquet(path):
    """View and explore a Parquet file with basic options."""
    # Check if file exists
    if not os.path.exists(path):
        print(f"Error: File not found at {path}")
        return False
    
    try:
        # Load the parquet file
        print(f"Loading Parquet file: {path}...")
        df = pd.read_parquet(path)
        
        # Basic information
        print("\n===== BASIC INFORMATION =====")
        print(f"Rows: {len(df)}")
        print(f"Columns: {len(df.columns)}")
        print(f"Memory usage: {df.memory_usage(deep=True).sum() / (1024 * 1024):.2f} MB")
        
        # Column information
        print("\n===== COLUMNS =====")
        for i, col in enumerate(df.columns):
            dtype = str(df[col].dtype)
            try:
                n_unique = df[col].nunique()
            except:
                n_unique = "N/A"
            print(f"{i+1}. {col} - Type: {dtype}, Unique values: {n_unique}")
        
        # Data sample
        print(f"\n===== DATA SAMPLE (5 rows) =====")
        pd.set_option('display.max_columns', 10)  # Limit columns for readability
        pd.set_option('display.width', 1000)
        pd.set_option('display.max_colwidth', 30)  # Truncate long values
        
        # Select simple columns for display (avoid complex types)
        simple_cols = []
        for col in df.columns[:10]:  # Only try first 10 columns
            try:
                # Check if column has simple data types
                if df[col].dtype in ['int64', 'float64', 'bool', 'object', 'datetime64[ns]']:
                    simple_cols.append(col)
            except:
                continue
        
        if simple_cols:
            print(df[simple_cols].head())
        else:
            print("No simple columns found for display")
        
        # Print a few example rows with all columns
        print("\n===== ROW EXAMPLES =====")
        for i in range(min(3, len(df))):
            print(f"\nRow {i}:")
            row = df.iloc[i]
            for col in df.columns:
                value = row[col]
                # Format the value to make it readable
                if isinstance(value, (list, np.ndarray)):
                    value = f"Array/List with {len(value)} items"
                elif isinstance(value, dict):
                    value = f"Dict with {len(value)} keys: {', '.join(list(value.keys())[:3])}..."
                elif pd.isna(value):
                    value = "NULL"
                else:
                    value = str(value)
                    if len(value) > 50:
                        value = value[:47] + "..."
                print(f"  {col}: {value}")
        
        return True
    
    except Exception as e:
        print(f"Error loading or processing the Parquet file: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    # Use default path or provided argument
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = "data/processed/test_master.parquet"
    
    view_parquet(file_path)

if __name__ == "__main__":
    main()