import pandas as pd
import glob

# path where csv files are stored
files = glob.glob("data/raw/*.csv")

merged_df = None

for file in files:
    
    df = pd.read_csv(file)

    # rename columns
    keyword = df.columns[1]
    df = df.rename(columns={keyword: keyword.lower().replace(" ", "_")})

    if merged_df is None:
        merged_df = df
    else:
        merged_df = pd.merge(merged_df, df, on="Time", how="outer")

# sort by time
merged_df = merged_df.sort_values("Time")

# save dataset
merged_df.to_csv("data/processed/fashion_trend_dataset.csv", index=False)

print("Datasets merged successfully!")