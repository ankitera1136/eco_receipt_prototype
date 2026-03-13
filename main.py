from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
import json
import re
import os

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Groq client ───────────────────────────────────────────────────────────────
# Paste your Groq key below OR set env variable: export GROQ_API_KEY=your_key
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1",
)

# Best Groq vision model for receipt OCR + analysis
VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

PROMPT = """You are EcoReceipt AI. Analyze this grocery receipt image carefully.

For each grocery item found, calculate its carbon footprint using these emission factors (kg CO2 per kg of food):
Beef/Lamb/Goat: 27 | Pork: 7.6 | Chicken/Poultry: 6.9 | Farmed fish: 6.0 | Wild fish: 3.0
Cheese: 13.5 | Butter: 9.0 | Cow milk: 3.2 per litre | Eggs: 4.5 per dozen (0.375 per egg)
Yogurt: 2.2 | Oat milk: 0.9 per litre | Almond milk: 0.7 per litre | Soy milk: 0.98 per litre
Rice: 4.0 | Bread/pasta/wheat: 1.4 | Oats/cereal: 1.6 | Chocolate/cocoa: 4.8
Coffee: 2.9 | Tofu/tempeh: 2.0 | Lentils/beans/legumes: 0.9 | Nuts: 2.5
Vegetables: 0.5 | Root vegetables: 0.4 | Local fruit: 0.3 | Tropical/imported fruit: 0.8
Snacks/chips: 3.5 | Sugary drinks/juice: 0.6 per litre | Beer/wine: 1.1 per litre
Cooking oil: 3.3 per litre | Sugar: 1.8 | Processed/frozen food: 3.0

SCORING FORMULA:
score = max(0, min(100, round((1 - total_co2 / 84) * 100)))
Grades: 85-100=A | 70-84=B+ | 55-69=B | 40-54=C+ | 25-39=C | 10-24=D | 0-9=F

INSTRUCTIONS:
1. Extract ALL grocery items visible on the receipt
2. Estimate quantity in kg or litres based on package sizes if not shown
3. Calculate co2_kg = emission_factor x quantity_kg
4. Set impact: high if co2_kg > 5, medium if 1.5 to 5, low if less than 1.5
5. Only suggest alternatives for items with co2_kg > 1.5
6. Generate helpful eco badges like "Chose Plant Milk", "Low Meat", "Fresh Veg Shopper"

Return ONLY a valid JSON object. No explanation, no markdown, just raw JSON:
{
  "store": "store name or Unknown",
  "date": "date from receipt or today",
  "items": [
    {
      "name": "product name",
      "category": "food category",
      "quantity_kg": 0.5,
      "emission_factor": 13.5,
      "co2_kg": 6.75,
      "impact": "high"
    }
  ],
  "total_co2": 0.0,
  "score": 0,
  "grade": "C",
  "grade_description": "short description",
  "improvement_message": "one actionable tip",
  "alternatives": [
    {
      "original": "item name",
      "swap": "alternative product",
      "saving_kg": 0.0,
      "saving_pct": 0,
      "reason": "brief explanation"
    }
  ],
  "badges": ["badge1", "badge2"]
}"""


class ScanRequest(BaseModel):
    image_b64: str
    media_type: str = "image/jpeg"


@app.post("/api/analyze")
async def analyze(req: ScanRequest):
    try:
        response = client.chat.completions.create(
            model=VISION_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{req.media_type};base64,{req.image_b64}"
                            }
                        },
                        {
                            "type": "text",
                            "text": PROMPT
                        }
                    ]
                }
            ],
            max_tokens=4096,
            temperature=0.3,
        )

        raw_text = response.choices[0].message.content

        # Clean up any markdown fences if present
        raw_text = re.sub(r'```json|```', '', raw_text).strip()

        # Extract JSON object
        match = re.search(r'\{.*\}', raw_text, re.DOTALL)
        if not match:
            raise ValueError("No JSON found in model response")

        result = json.loads(match.group())
        return result

    except json.JSONDecodeError as e:
        raise HTTPException(status_code=422, detail=f"Could not parse receipt. Try a clearer image. ({str(e)})")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Serve frontend at http://localhost:8000
app.mount("/", StaticFiles(directory="static", html=True), name="static")