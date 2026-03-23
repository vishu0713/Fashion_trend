import pandas as pd
import numpy as np
import os
import re

# -----------------------------
# 1. File paths
# -----------------------------
INPUT_FILE = "data/processed/fashion_trend_dataset.csv"
OUTPUT_FILE = "data/processed/fashion_trend_features.csv"

# -----------------------------
# 2. Helper function to clean column names
# -----------------------------
def clean_column_name(col_name: str) -> str:
    col_name = col_name.strip().lower()
    col_name = re.sub(r"[^\w\s]", "", col_name)   # remove special characters
    col_name = re.sub(r"\s+", "_", col_name)      # replace spaces with underscore
    return col_name

# -----------------------------
# 3. Load dataset
# -----------------------------
df = pd.read_csv(INPUT_FILE)

# Clean all column names
df.columns = [clean_column_name(col) for col in df.columns]

# Rename first column to date if needed
if df.columns[0] != "date":
    df = df.rename(columns={df.columns[0]: "date"})

# Convert date column
df["date"] = pd.to_datetime(df["date"], errors="coerce", dayfirst=True)

# Drop rows where date is invalid
df = df.dropna(subset=["date"])

# Sort by date
df = df.sort_values("date").reset_index(drop=True)

# -----------------------------
# 4. Identify keyword columns
# -----------------------------
keyword_columns = [col for col in df.columns if col != "date"]

all_feature_rows = []

# -----------------------------
# 5. Create ML rows for each keyword
# -----------------------------
for keyword in keyword_columns:
    temp = df[["date", keyword]].copy()
    temp = temp.rename(columns={keyword: "trend"})
    
    # Remove missing values
    temp = temp.dropna()

    # Create lag features
    temp["lag1"] = temp["trend"].shift(1)
    temp["lag2"] = temp["trend"].shift(2)
    temp["lag3"] = temp["trend"].shift(3)

    # Rolling features
    temp["ma3"] = temp["trend"].rolling(window=3).mean()
    temp["ma6"] = temp["trend"].rolling(window=6).mean()
    temp["std3"] = temp["trend"].rolling(window=3).std()

    # Trend change / slope
    temp["slope"] = temp["trend"] - temp["lag1"]
    temp["pct_change"] = temp["trend"].pct_change()

    # Add keyword column
    temp["keyword"] = keyword

    # Create classification target
    # 1 if next value is greater than current value, else 0
    temp["target"] = (temp["trend"].shift(-1) > temp["trend"]).astype(int)

    # Drop rows with NaN created by shifting/rolling
    temp = temp.dropna()

    all_feature_rows.append(temp)

# -----------------------------
# 6. Combine all keywords
# -----------------------------
final_df = pd.concat(all_feature_rows, ignore_index=True)

# Reorder columns
final_df = final_df[
    [
        "date",
        "keyword",
        "trend",
        "lag1",
        "lag2",
        "lag3",
        "ma3",
        "ma6",
        "std3",
        "slope",
        "pct_change",
        "target",
    ]
]

# -----------------------------
# 7. Save output
# -----------------------------
os.makedirs("data/processed", exist_ok=True)
final_df.to_csv(OUTPUT_FILE, index=False)

print("Feature engineering completed successfully.")
print(f"Saved file: {OUTPUT_FILE}")
print(final_df.head())
print(f"Total rows: {len(final_df)}")