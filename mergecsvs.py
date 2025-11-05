
import os
import pandas as pd
import re
from glob import glob

# ---- CONFIG ----
DATA_DIR = "/Users/anushajain/Desktop/google_genai_hack/data"      
OUTPUT_PATH = "merged_12_months_hmis.csv"   
DISTRICT_NAME = "Solapur"         

# ---- HELPER FUNCTIONS ----
def normalize_columns(df):
    """Standardize HMIS column names across months."""
    rename_map = {
        "Sub-District": "Sub District",
        "SubDistrict": "Sub District",
        "Facility_Type": "Facility Type",
        "Facility_Name": "Facility Name",
        "Item": "Item Name",
        "Indicator": "Item Name",
        "Category_Name": "Category",
        "Indicator_Name": "Item Name",
        "Month_Name": "Month",
        "Month of Report": "Month",
        "Value (Number)": "Value"
    }
    df = df.rename(columns=rename_map)
    return df

def extract_month_from_filename(filename):
    """Try to extract month name from filename (e.g., 'May', 'Jan-2018')."""
    fname = os.path.basename(filename)
    match = re.search(r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)", fname, re.IGNORECASE)
    return match.group(1).title() if match else "Unknown"

# ---- MAIN ----
def main():
    all_files = glob(os.path.join(DATA_DIR, "*.csv"))
    if not all_files:
        print("❌ No CSVs found in:", DATA_DIR)
        return

    dfs = []
    for path in all_files:
        print(f"📥 Reading {path} ...")
        try:
            df = pd.read_csv(path, encoding="utf-8")
        except Exception:
            df = pd.read_csv(path, encoding="latin-1")

        df = normalize_columns(df)
        df["Month_File"] = extract_month_from_filename(path)
        df["District"] = DISTRICT_NAME
        df["Source_File"] = os.path.basename(path)
        dfs.append(df)

    merged = pd.concat(dfs, ignore_index=True, sort=False)
    merged.to_csv(OUTPUT_PATH, index=False)
    print(f"✅ Merged {len(all_files)} files → {OUTPUT_PATH}")
    print(f"📊 Total rows: {len(merged)}  |  Columns: {len(merged.columns)}")

if __name__ == "__main__":
    main()
