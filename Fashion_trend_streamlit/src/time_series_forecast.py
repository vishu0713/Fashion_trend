import pandas as pd
import re
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import matplotlib.pyplot as plt

DATA_FILE = "data/processed/fashion_trend_dataset.csv"


def clean_column_name(col_name: str) -> str:
    col_name = col_name.strip().lower()
    col_name = re.sub(r"[^\w\s]", "", col_name)
    col_name = re.sub(r"\s+", "_", col_name)
    return col_name


# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_csv("C:/Users/vishu/OneDrive/Desktop/Fashion_trend/data/processed/fashion_trend_dataset.csv")
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


def plot_forecast(result):
    if not result["success"]:
        print(result["message"])
        return

    history = result["history"]
    forecast_df = result["forecast_df"]

    plt.figure(figsize=(12, 6))

    plt.plot(
        history["date"],
        history["trend"],
        marker="o",
        linewidth=2,
        label="Historical Trend"
    )

    plt.plot(
        forecast_df["date"],
        forecast_df["forecast"],
        marker="o",
        linestyle="--",
        linewidth=2,
        label="Forecast"
    )

    plt.title(f"Forecast for {result['keyword'].replace('_', ' ').title()}")
    plt.xlabel("Date")
    plt.ylabel("Trend Score")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    user_keyword = "cargo pants"
    forecast_steps = 3

    result = forecast_keyword(user_keyword, steps=forecast_steps)

    if result["success"]:
        print("\nForecast Result")
        print("-" * 40)
        print("Keyword:", result["keyword"])
        print("Next", forecast_steps, "month(s) forecast:")
        print(result["forecast_df"])
        print("-" * 40)

        plot_forecast(result)
    else:
        print(result["message"])