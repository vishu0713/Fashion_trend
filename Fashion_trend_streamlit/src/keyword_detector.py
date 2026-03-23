import re

FASHION_KEYWORDS = [
    "cargo pants",
    "oversized hoodie",
    "baggy jeans",
    "puffer jacket",
    "linen shirt",
    "leather jacket",
    "mini skirt",
    "chunky sneakers"
    "oversized_blazer"
    "pleated skirt"
    "maxi skirt"
    "white sneakers"
    "platform sneakers"
    "nike dunk"
    "adidas samba"
    "combat boots"
    "wide_leg_jeans"
    "parachute pants"
    "cropped hoodie"
    "graphic hoodie"
    "corset top"
    "tank top"
    "denim jacket"
    "bomber jacket"
]


def detect_keyword(user_text: str):
    text = user_text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)

    for keyword in FASHION_KEYWORDS:
        if keyword in text:
            return keyword

    return None


if __name__ == "__main__":
    user_query = "Tell me about cargo pants"
    print(detect_keyword(user_query))