import pandas as pd
import re
import joblib
import matplotlib.pyplot as plt

DATA_FILE = "data/processed/fashion_trend_dataset.csv"
MODEL_FILE = "models/trend_classifier.pkl"
ENCODER_FILE = "models/keyword_encoder.pkl"
SCALER_FILE = "models/scaler.pkl"


def clean_column_name(col_name: str) -> str:
    col_name = col_name.strip().lower()
    col_name = re.sub(r"[^\w\s]", "", col_name)
    col_name = re.sub(r"\s+", "_", col_name)
    return col_name


# -----------------------------
# Load data
# -----------------------------
df = pd.read_csv(DATA_FILE)
df.columns = [clean_column_name(col) for col in df.columns]

if df.columns[0] != "date":
    df = df.rename(columns={df.columns[0]: "date"})

df["date"] = pd.to_datetime(df["date"], errors="coerce", dayfirst=True)
df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

# -----------------------------
# Load model files
# -----------------------------
model = joblib.load(MODEL_FILE)
keyword_encoder = joblib.load(ENCODER_FILE)
scaler = joblib.load(SCALER_FILE)


def analyze_keyword_trend(keyword: str):
    keyword = clean_column_name(keyword)

    if keyword not in df.columns:
        return {
            "success": False,
            "message": f"Keyword '{keyword}' not found in dataset."
        }

    temp = df[["date", keyword]].copy()
    temp = temp.rename(columns={keyword: "trend"})
    temp = temp.dropna().reset_index(drop=True)

    if len(temp) < 6:
        return {
            "success": False,
            "message": f"Not enough data available for '{keyword}'."
        }

    latest_trend = temp["trend"].iloc[-1]
    lag1 = temp["trend"].iloc[-2]
    lag2 = temp["trend"].iloc[-3]
    lag3 = temp["trend"].iloc[-4]

    ma3 = temp["trend"].iloc[-3:].mean()
    ma6 = temp["trend"].iloc[-6:].mean()
    std3 = temp["trend"].iloc[-3:].std()
    slope = latest_trend - lag1
    pct_change = 0 if lag1 == 0 else (latest_trend - lag1) / lag1

    try:
        keyword_encoded = keyword_encoder.transform([keyword])[0]
    except ValueError:
        return {
            "success": False,
            "message": f"Keyword '{keyword}' is not available in trained encoder."
        }

    feature_df = pd.DataFrame([{
        "keyword_encoded": keyword_encoded,
        "trend": latest_trend,
        "lag1": lag1,
        "lag2": lag2,
        "lag3": lag3,
        "ma3": ma3,
        "ma6": ma6,
        "std3": std3,
        "slope": slope,
        "pct_change": pct_change
    }])

    if type(model).__name__ == "LogisticRegression":
        X_input = scaler.transform(feature_df)
        prediction = model.predict(X_input)[0]
        probability = model.predict_proba(X_input)[0][1]
    else:
        prediction = model.predict(feature_df)[0]
        probability = model.predict_proba(feature_df)[0][1]

    recent_direction = "Rising" if slope > 0 else "Falling" if slope < 0 else "Stable"

    return {
        "success": True,
        "keyword": keyword,
        "latest_date": str(temp["date"].iloc[-1].date()),
        "latest_trend": float(latest_trend),
        "previous_trend": float(lag1),
        "ma3": float(ma3),
        "ma6": float(ma6),
        "std3": float(std3),
        "slope": float(slope),
        "pct_change": float(pct_change),
        "recent_direction": recent_direction,
        "prediction": int(prediction),
        "prediction_label": "Upward" if prediction == 1 else "Not Upward",
        "confidence": float(round(probability, 4)),
        "chart_data": temp.tail(24).copy()
    }


def plot_trend_analysis(result):
    if not result["success"]:
        print(result["message"])
        return

    chart_df = result["chart_data"]

    plt.figure(figsize=(12, 6))
    plt.plot(chart_df["date"], chart_df["trend"], marker="o", linewidth=2)
    plt.scatter(chart_df["date"].iloc[-1], chart_df["trend"].iloc[-1], s=100)

    plt.title(f"Trend Analysis for {result['keyword'].replace('_', ' ').title()}")
    plt.xlabel("Date")
    plt.ylabel("Trend Score")
    plt.xticks(rotation=45)
    plt.grid(True)

    info_text = (
        f"Prediction: {result['prediction_label']}\n"
        f"Confidence: {result['confidence']}\n"
        f"Recent Direction: {result['recent_direction']}"
    )

    plt.text(
        0.02, 0.95,
        info_text,
        transform=plt.gca().transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
    )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    user_keyword = "cargo pants"

    result = analyze_keyword_trend(user_keyword)

    if result["success"]:
        print("\nTrend Analysis Result")
        print("-" * 40)
        print("Keyword:", result["keyword"])
        print("Latest Date:", result["latest_date"])
        print("Latest Trend:", result["latest_trend"])
        print("Previous Trend:", result["previous_trend"])
        print("3-Period Moving Avg:", round(result["ma3"], 2))
        print("6-Period Moving Avg:", round(result["ma6"], 2))
        print("Recent Direction:", result["recent_direction"])
        print("Prediction:", result["prediction_label"])
        print("Confidence:", result["confidence"])
        print("-" * 40)

        plot_trend_analysis(result)
    else:
        print(result["message"])