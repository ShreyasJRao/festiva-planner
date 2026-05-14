from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from functools import lru_cache

app = FastAPI(title="Festiva Planner AI")

@lru_cache(maxsize=1)
def load_models():
    nlp_model = joblib.load('../models/nlp_model.pkl')
    vectorizer = joblib.load('../models/vectorizer.pkl')
    budget_model = joblib.load('../models/budget_model.pkl')
    knowledge_base = joblib.load('../models/knowledge_base.pkl')
    index = faiss.read_index('../models/knowledge_index.faiss')
    embed_model = SentenceTransformer('all-MiniLM-L6-v2',
                  cache_folder='../models/sentence_transformer_cache')
    return nlp_model, vectorizer, budget_model, knowledge_base, index, embed_model

CITY_MULTIPLIERS = {
    'Mumbai': 1.5, 'Delhi': 1.4, 'Bangalore': 1.3, 'Hyderabad': 1.1,
    'Chennai': 1.1, 'Pune': 1.2, 'Kolkata': 1.1, 'Ahmedabad': 1.0,
    'Jaipur': 1.0, 'Surat': 0.9, 'Lucknow': 0.9, 'Kanpur': 0.85,
    'Nagpur': 0.9, 'Indore': 0.9, 'Bhopal': 0.85, 'Visakhapatnam': 0.9,
    'Patna': 0.8, 'Vadodara': 0.9, 'Ghaziabad': 1.0, 'Coimbatore': 0.9,
    'Kochi': 1.0, 'Chandigarh': 1.1, 'Goa': 1.3, 'Mysore': 1.0
}

BASE_BUDGETS = {
    'Wedding': 1500000,
    'Corporate': 800000,
    'Birthday': 200000,
    'Anniversary': 350000,
    'Party': 150000
}

BUDGET_WARNINGS = {
    'Wedding':     {'per_guest': 3500, 'min': 300000},
    'Corporate':   {'per_guest': 2500, 'min': 150000},
    'Birthday':    {'per_guest': 1500, 'min': 50000},
    'Anniversary': {'per_guest': 2000, 'min': 80000},
    'Party':       {'per_guest': 1200, 'min': 40000},
}

SPLITS = {
    'Wedding':     {'Venue': 0.30, 'Catering': 0.25, 'Decoration': 0.20, 'Photography': 0.15, 'Miscellaneous': 0.10},
    'Corporate':   {'Venue': 0.40, 'Catering': 0.30, 'AV Equipment': 0.15, 'Miscellaneous': 0.15},
    'Birthday':    {'Venue': 0.35, 'Catering': 0.30, 'Decoration': 0.20, 'Cake': 0.10, 'Entertainment': 0.05},
    'Anniversary': {'Venue': 0.50, 'Decoration': 0.25, 'Gifts': 0.15, 'Photography': 0.10},
    'Party':       {'Venue': 0.35, 'Catering': 0.35, 'Decoration': 0.20, 'Entertainment': 0.10},
}

CHECKLISTS = {
    'Wedding': [
        '12 months before: Book venue',
        '8 months before: Finalize catering',
        '6 months before: Book photographer',
        '3 months before: Send invitations',
        '1 month before: Confirm all vendors',
        '1 week before: Final headcount',
    ],
    'Corporate': [
        '3 months before: Book venue and AV equipment',
        '2 months before: Finalize agenda and speakers',
        '1 month before: Send invitations',
        '2 weeks before: Confirm catering',
        '1 week before: Technical rehearsal',
    ],
    'Birthday': [
        '1 month before: Book venue',
        '3 weeks before: Send invitations',
        '1 week before: Order cake and decorations',
        '1 day before: Setup venue',
        'Day of: Arrange return gifts',
    ],
    'Anniversary': [
        '2 months before: Book venue',
        '1 month before: Send invitations',
        '1 week before: Arrange flowers and cake',
        '1 day before: Setup decorations',
        'Day of: Coordinate with photographer',
    ],
    'Party': [
        '1 month before: Decide venue and theme',
        '3 weeks before: Send invitations',
        '1 week before: Arrange food and drinks',
        '1 day before: Setup decorations',
        'Day of: Arrange music and entertainment',
    ],
}

VENDORS = {
    'Wedding':     [('Photographer', 0.15), ('Caterer', 0.25), ('Decorator', 0.20), ('DJ/Band', 0.05), ('Mehendi Artist', 0.03), ('Florist', 0.05)],
    'Corporate':   [('AV Equipment', 0.15), ('Caterer', 0.30), ('Event MC', 0.05), ('Photographer', 0.08), ('Transport', 0.05)],
    'Birthday':    [('Caterer', 0.30), ('Decorator', 0.20), ('Cake Shop', 0.10), ('Entertainer', 0.05), ('Photographer', 0.08)],
    'Anniversary': [('Florist', 0.10), ('Caterer', 0.25), ('Photographer', 0.10), ('Decorator', 0.25), ('Musician', 0.05)],
    'Party':       [('Caterer', 0.35), ('DJ', 0.10), ('Decorator', 0.20), ('Photographer', 0.08), ('Bartender', 0.05)],
}

class EventRequest(BaseModel):
    event_type: str
    preferences: str
    city: str
    guest_count: int
    duration_days: int
    season: str
    outdoor: bool
    user_budget: int

@app.get("/")
def root():
    return {"message": "Festiva Planner AI is running!"}

@app.get("/cities")
def get_cities():
    return {"cities": sorted(CITY_MULTIPLIERS.keys())}

@app.post("/plan")
def generate_plan(request: EventRequest):
    _, _, _, knowledge_base, index, embed_model = load_models()

    event_type = request.event_type
    city_mult = CITY_MULTIPLIERS.get(request.city, 1.0)

    # Smart budget logic
    warning = None
    min_required = (
        BUDGET_WARNINGS[event_type]['per_guest'] * request.guest_count +
        BUDGET_WARNINGS[event_type]['min']
    ) * city_mult

    if request.user_budget < min_required * 0.5:
        warning = f"Your budget of ₹{request.user_budget:,} is very low for {request.guest_count} guests in {request.city}. Minimum recommended is ₹{int(min_required):,}. Plan adjusted to minimum viable budget."
        working_budget = int(min_required * 0.5)
    elif request.user_budget < min_required:
        warning = f"Your budget of ₹{request.user_budget:,} is a bit tight for {request.guest_count} guests in {request.city}. We recommend at least ₹{int(min_required):,} for a comfortable event. We've created the best plan possible within your budget."
        working_budget = request.user_budget
    else:
        working_budget = request.user_budget

    # Budget breakdown
    breakdown = {k: int(working_budget * v) for k, v in SPLITS[event_type].items()}

    # Vendor suggestions with estimated costs
    vendors = [
        {"name": name, "estimated_cost": int(working_budget * pct)}
        for name, pct in VENDORS[event_type]
    ]

    # RAG knowledge
    query_vector = embed_model.encode([f"{event_type} planning tips {request.preferences}"])
    distances, indices = index.search(np.array(query_vector), 2)
    tips = [knowledge_base[i] for i in indices[0]]

    return {
        "event_type": event_type,
        "city": request.city,
        "guest_count": request.guest_count,
        "duration_days": request.duration_days,
        "user_budget": request.user_budget,
        "working_budget": working_budget,
        "warning": warning,
        "budget_breakdown": breakdown,
        "checklist": CHECKLISTS[event_type],
        "vendors": vendors,
        "tips": tips,
    }