import json
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(BASE_DIR, "helpers.json")) as f:
    helpers = json.load(f)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

POLICY_TEXT = """
Sahay Safety Policy:
- A helper must be verified to be dispatched for medical or mobility bookings.
- Blacklisted helpers are never dispatched under any circumstance.
- Helpers with more than 2 no-shows are flagged as lower reliability.
- Same-zone helpers are preferred to reduce response time.
"""

class BookingRequest(BaseModel):
    sector_needed: str
    zone: str
    urgency: str = "medium"

def trust_score(h):
    if h["blacklisted"]:
        return 0.0
    score = h["rating_avg"] * 20
    if h["verified"]:
        score += 10
    score -= h["no_show_count"] * 5
    return max(score, 0)

def same_area_bonus(h, booking):
    return 15 if h["zone"] == booking["zone"] else 0

def sector_fit(h, booking):
    return 20 if h["sector"] == booking["sector_needed"] else 0

def match_score(h, booking):
    return trust_score(h) + same_area_bonus(h, booking) + sector_fit(h, booking)

def rank_helpers(booking, top_n=3):
    scored = []
    for h in helpers:
        if not h.get("available", True):
            continue
        if sector_fit(h, booking) == 0:
            continue
        scored.append((match_score(h, booking), h))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:top_n]

def explain_match(booking, top_helper):
    if not client:
        return "AI explanation unavailable (missing API key)."
    prompt = f"""
You are Sahay's matching assistant. Using the policy below, explain in one
friendly sentence why this helper was matched for this booking. Mention
verification status, rating, and zone match. Be concise.

Policy:
{POLICY_TEXT}

Booking: {booking['sector_needed']} needed in {booking['zone']}, urgency={booking['urgency']}
Matched helper: {top_helper['name']}, verified={top_helper['verified']},
rating={top_helper['rating_avg']}, zone={top_helper['zone']},
no_shows={top_helper['no_show_count']}

Explanation:
"""
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )
    return response.text.strip()

@app.get("/")
def root():
    return {"status": "Sahay backend is running"}

@app.post("/match")
def match(booking: BookingRequest):
    booking_dict = booking.dict()
    ranked = rank_helpers(booking_dict)
    if not ranked:
        return {"matches": [], "message": "No available helpers for this sector."}

    results = []
    for score, h in ranked:
        results.append({
            "helper_id": h["helper_id"],
            "name": h["name"],
            "zone": h["zone"],
            "rating": h["rating_avg"],
            "verified": h["verified"],
            "score": round(score, 1)
        })

    explanation = explain_match(booking_dict, ranked[0][1])

    return {
        "matches": results,
        "top_match_explanation": explanation
    }
