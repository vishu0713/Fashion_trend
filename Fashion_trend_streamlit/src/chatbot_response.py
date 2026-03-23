import os
from pathlib import Path
from google import genai
from dotenv import load_dotenv

# Load .env from project root
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)


def generate_fashion_response(
    user_query: str,
    keyword: str,
    latest_trend: float,
    previous_trend: float,
    recent_direction: str,
    prediction_label: str,
    confidence: float,
    forecast_values: list[float],
) -> str:

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        return "❌ Gemini API key not found. Please set GEMINI_API_KEY in your .env file."

    # Create Gemini client
    client = genai.Client(api_key=api_key)

    prompt = f"""
You are a fashion trend assistant.

User question: "{user_query}"

Detected item: {keyword}

Trend info:
- Latest score: {latest_trend}
- Previous score: {previous_trend}
- Direction: {recent_direction}
- Prediction: {prediction_label}
- Confidence: {confidence}
- Forecast (next months): {forecast_values}

Instructions:
- Be friendly and stylish like a fashion advisor
- Clearly say if the trend is rising, stable, or falling
- Mention future trend briefly
- Keep it SHORT (3–5 lines max)
- Do NOT make up numbers
"""

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        if hasattr(response, "text") and response.text:
            return response.text.strip()

        return "⚠️ Gemini returned an empty response."

    except Exception as e:
        return f"❌ Error generating Gemini response: {str(e)}"


# Test run
if __name__ == "__main__":
    response = generate_fashion_response(
        user_query="Is cargo pants trending?",
        keyword="cargo pants",
        latest_trend=32,
        previous_trend=29,
        recent_direction="Rising",
        prediction_label="Upward",
        confidence=0.78,
        forecast_values=[34, 36, 38],
    )

    print("\n💬 Chatbot Response:\n")
    print(response)