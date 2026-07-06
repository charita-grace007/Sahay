 · PY
import json
import os
import uuid
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
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
 
# Make sure every helper has the new fields (backwards-compatible if old json is missing them)
for h in helpers:
    h.setdefault("is_student", False)
    h.setdefault("credits_earned", 0)
    h.setdefault("reviews", [])  # list of {rating, comment, created_at}
    h.setdefault("bio", "")
    h.setdefault("photo_url", "")
 
# In-memory bookings store (resets on server restart, fine for demo)
bookings = {}
 
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
    requester_name: str = "Guest"
 
 
class ReviewRequest(BaseModel):
    rating: float
    comment: str = ""
 
 
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
 
 
def helper_public(h):
    """Return a helper dict safe/ready for frontend display."""
    return {
        "helper_id": h["helper_id"],
        "name": h["name"],
        "zone": h["zone"],
        "sector": h["sector"],
        "rating": h["rating_avg"],
        "verified": h["verified"],
        "is_student": h["is_student"],
        "bio": h["bio"],
        "photo_url": h["photo_url"],
        "reviews": h["reviews"][-5:],  # last 5 reviews
    }
 
 
@app.get("/")
def root():
    return {"status": "Sahay backend is running"}
 
 
@app.get("/sectors")
def get_sectors():
    return {"sectors": sorted(list(set(h["sector"] for h in helpers)))}
 
 
@app.get("/zones")
def get_zones():
    return {"zones": sorted(list(set(h["zone"] for h in helpers)))}
 
 
@app.post("/match")
def match(booking: BookingRequest):
    booking_dict = booking.dict()
    ranked = rank_helpers(booking_dict)
    if not ranked:
        return {"matches": [], "message": "No available helpers for this sector."}
 
    results = []
    for score, h in ranked:
        entry = helper_public(h)
        entry["score"] = round(score, 1)
        results.append(entry)
 
    explanation = explain_match(booking_dict, ranked[0][1])
 
    # Create a pending booking against the top match
    top_helper = ranked[0][1]
    booking_id = str(uuid.uuid4())[:8]
    bookings[booking_id] = {
        "booking_id": booking_id,
        "requester_name": booking.requester_name,
        "sector_needed": booking.sector_needed,
        "zone": booking.zone,
        "urgency": booking.urgency,
        "helper_id": top_helper["helper_id"],
        "helper_name": top_helper["name"],
        "status": "pending",
        "created_at": datetime.utcnow().isoformat(),
    }
 
    return {
        "booking_id": booking_id,
        "matches": results,
        "top_match_explanation": explanation
    }
 
 
@app.post("/bookings/{booking_id}/accept")
def accept_booking(booking_id: str):
    b = bookings.get(booking_id)
    if not b:
        raise HTTPException(status_code=404, detail="Booking not found")
    b["status"] = "accepted"
    return b
 
 
@app.post("/bookings/{booking_id}/complete")
def complete_booking(booking_id: str):
    b = bookings.get(booking_id)
    if not b:
        raise HTTPException(status_code=404, detail="Booking not found")
    b["status"] = "completed"
    return b
 
 
@app.get("/bookings/{booking_id}")
def get_booking(booking_id: str):
    b = bookings.get(booking_id)
    if not b:
        raise HTTPException(status_code=404, detail="Booking not found")
    return b
 
 
@app.get("/bookings")
def list_bookings(requester_name: Optional[str] = None):
    vals = list(bookings.values())
    if requester_name:
        vals = [b for b in vals if b["requester_name"] == requester_name]
    return {"bookings": vals}
 
 
@app.post("/bookings/{booking_id}/review")
def review_booking(booking_id: str, review: ReviewRequest):
    b = bookings.get(booking_id)
    if not b:
        raise HTTPException(status_code=404, detail="Booking not found")
    if b["status"] != "completed":
        raise HTTPException(status_code=400, detail="Booking must be completed before review")
 
    helper = next((h for h in helpers if h["helper_id"] == b["helper_id"]), None)
    if not helper:
        raise HTTPException(status_code=404, detail="Helper not found")
 
    helper["reviews"].append({
        "rating": review.rating,
        "comment": review.comment,
        "created_at": datetime.utcnow().isoformat(),
    })
 
    # Recalculate rating average from reviews
    all_ratings = [r["rating"] for r in helper["reviews"]]
    helper["rating_avg"] = round(sum(all_ratings) / len(all_ratings), 2)
 
    # Reward: students earn credits for completed + reviewed bookings
    credits_awarded = 0
    if helper["is_student"]:
        credits_awarded = 10 if review.rating >= 4 else 5
        helper["credits_earned"] += credits_awarded
 
    b["status"] = "reviewed"
    return {
        "helper_id": helper["helper_id"],
        "new_rating_avg": helper["rating_avg"],
        "credits_awarded": credits_awarded,
        "total_credits": helper["credits_earned"],
    }
 
