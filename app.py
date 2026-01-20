# app.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware  # <--- NEW IMPORT
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F

app = FastAPI(title="HallucinoGen API")

# --- NEW BLOCK: ALLOW BROWSER CONNECTIONS (CORS) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (Chrome extension, local files, etc.)
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (POST, GET, OPTIONS, etc.)
    allow_headers=["*"],
)
# ---------------------------------------------------

# Load Model (Using your local folder)
MODEL_NAME = "./my_hallucinogen_model"
print("Loading Model...")
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)
    model.eval()
except Exception as e:
    print(f"Error loading model: {e}")
    print("Ensure 'my_hallucinogen_model' folder is in the same directory as app.py")


class CheckRequest(BaseModel):
    premise: str
    hypothesis: str


@app.post("/check")
async def check_hallucination(request: CheckRequest):
    try:
        inputs = tokenizer(
            request.premise,
            request.hypothesis,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )

        with torch.no_grad():
            outputs = model(**inputs)

        probs = F.softmax(outputs.logits, dim=-1)[0].tolist()
        hallucination_score = probs[2]
        is_hallucination = hallucination_score > 0.30

        return {
            "status": "Hallucination" if is_hallucination else "Safe",
            "hallucination_score": round(hallucination_score, 4),
            "confidence": round(max(probs), 4)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))