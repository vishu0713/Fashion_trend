import pandas as pd
import re
from pathlib import Path
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "data" / "processed" / "fashion_trend_dataset.csv"


def clean_column_name(col_name: str) -> str:
    col_name = col_name.strip().lower()
    col_name = re.sub(r"[^\w\s]", "", col_name)
    col_name = re.sub(r"\s+", "_", col_name)
    return col_name


df = pd.read_csv(DATA_FILE)
df.columns = [clean_column_name(col) for col in df.columns]

if df.columns[0] != "date":
    df = df.rename(columns={df.columns[0]: "date"})

df["date"] = pd.to_datetime(df["date"], errors="coerce", dayfirst=True)
df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)


def forecast_keyword(keyword: str, steps: int = 3):
    keyword = clean_column_name(keyword)

    if keyword not in df.columns:
        return {
            "success": False,
            "message": f"Keyword '{keyword}' not found in dataset."
        }

    series = df[["date", keyword]].copy()
    series = series.rename(columns={keyword: "trend"}).dropna().reset_index(drop=True)

    if len(series) < 8:
        return {
            "success": False,
            "message": f"Not enough data to forecast '{keyword}'."
        }

    model = ExponentialSmoothing(
        series["trend"],
        trend="add",
        seasonal=None
    ).fit()

    forecast_values = model.forecast(steps)

    last_date = series["date"].iloc[-1]
    forecast_dates = pd.date_range(
        start=last_date + pd.DateOffset(months=1),
        periods=steps,
        freq="MS"
    )

    forecast_df = pd.DataFrame({
        "date": forecast_dates,
        "forecast": forecast_values.values
    })

    return {
        "success": True,
        "keyword": keyword,
        "history": series.tail(24).copy(),
        "forecast_df": forecast_df
    }