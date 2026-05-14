import streamlit as st
import joblib
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from functools import lru_cache

st.set_page_config(page_title="Festiva Planner AI", page_icon="🎉", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
* { font-family: 'Inter', sans-serif; }
.stApp { background-color: #0a0a0f; }
.hero { text-align: center; padding: 48px 0 32px; }
.hero-badge { display: inline-block; background: #1a1a2e; color: #a78bfa; border: 1px solid #4c1d95; padding: 6px 16px; border-radius: 20px; font-size: 12px; letter-spacing: 1px; text-transform: uppercase; margin-bottom: 20px; }
.hero-title { font-size: 52px; font-weight: 700; background: linear-gradient(135deg, #ffffff 0%, #a78bfa 50%, #6d28d9 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; line-height: 1.1; margin-bottom: 16px; }
.hero-sub { color: #475569; font-size: 18px; max-width: 500px; margin: 0 auto 40px; line-height: 1.6; }
.form-card { background: #111118; border: 1px solid #1e1e2e; border-radius: 20px; padding: 32px; margin-bottom: 24px; }
.form-section-title { color: #64748b; font-size: 11px; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 16px; }
.result-header { background: #111118; border: 1px solid #1e1e2e; border-radius: 20px; padding: 28px; margin-bottom: 20px; }
.event-type-pill { display: inline-block; background: #2d1b69; color: #a78bfa; border: 1px solid #4c1d95; padding: 8px 20px; border-radius: 24px; font-size: 13px; font-weight: 600; letter-spacing: 0.5px; margin-bottom: 16px; }
.big-budget { font-size: 48px; font-weight: 700; color: #a78bfa; line-height: 1; margin-bottom: 8px; }
.budget-label { color: #475569; font-size: 14px; }
.metric-row { display: flex; gap: 12px; margin-top: 24px; flex-wrap: wrap; }
.metric-box { flex: 1; min-width: 80px; background: #0a0a0f; border: 1px solid #1e1e2e; border-radius: 12px; padding: 14px; }
.metric-val { color: #e2e8f0; font-size: 18px; font-weight: 600; }
.metric-lbl { color: #475569; font-size: 11px; margin-top: 4px; text-transform: uppercase; letter-spacing: 0.5px; }
.section-card { background: #111118; border: 1px solid #1e1e2e; border-radius: 16px; padding: 24px; margin-bottom: 16px; }
.section-head { color: #a78bfa; font-size: 11px; text-transform: uppercase; letter-spacing: 1.5px; font-weight: 600; margin-bottom: 16px; }
.check-item { display: flex; align-items: flex-start; gap: 12px; padding: 10px 0; border-bottom: 1px solid #0f0f18; }
.check-num { background: #2d1b69; color: #a78bfa; width: 24px; height: 24px; border-radius: 6px; display: flex; align-items: center; justify-content: center; font-size: 11px; font-weight: 600; flex-shrink: 0; }
.check-text { color: #94a3b8; font-size: 14px; line-height: 1.5; padding-top: 2px; }
.tip-card { background: #0a0a0f; border-left: 3px solid #7c3aed; border-radius: 0 10px 10px 0; padding: 14px 16px; margin-bottom: 10px; color: #94a3b8; font-size: 13px; line-height: 1.7; }
.warning-box { background: #1a1200; border: 1px solid #854d0e; border-radius: 12px; padding: 16px 20px; margin-bottom: 20px; color: #fbbf24; font-size: 14px; line-height: 1.6; }
.stButton > button { background: linear-gradient(135deg, #7c3aed, #6d28d9) !important; color: white !important; border: none !important; border-radius: 12px !important; padding: 16px 28px !important; font-size: 16px !important; font-weight: 600 !important; width: 100% !important; }
.reset-btn > button { background: #111118 !important; color: #94a3b8 !important; border: 1px solid #1e1e2e !important; border-radius: 12px !important; padding: 12px 28px !important; font-size: 14px !important; width: 100% !important; }
div[data-testid="stSelectbox"] > div, div[data-testid="stNumberInput"] > div > div, div[data-testid="stTextInput"] > div > div { background: #0a0a0f !important; border-color: #1e1e2e !important; color: #e2e8f0 !important; border-radius: 10px !important; }
label { color: #64748b !important; font-size: 12px !important; text-transform: uppercase !important; letter-spacing: 0.5px !important; }
footer { visibility: hidden; }
#MainMenu { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── DATA ──────────────────────────────────────────────────
CITIES = sorted(['Mumbai','Delhi','Bangalore','Hyderabad','Chennai','Pune',
    'Kolkata','Ahmedabad','Jaipur','Surat','Lucknow','Kanpur','Nagpur',
    'Indore','Bhopal','Visakhapatnam','Patna','Vadodara','Ghaziabad',
    'Coimbatore','Kochi','Chandigarh','Goa','Mysore'])

CITY_MULT = {'Mumbai':1.5,'Delhi':1.4,'Bangalore':1.3,'Hyderabad':1.1,
    'Chennai':1.1,'Pune':1.2,'Kolkata':1.1,'Ahmedabad':1.0,'Jaipur':1.0,
    'Surat':0.9,'Lucknow':0.9,'Kanpur':0.85,'Nagpur':0.9,'Indore':0.9,
    'Bhopal':0.85,'Visakhapatnam':0.9,'Patna':0.8,'Vadodara':0.9,
    'Ghaziabad':1.0,'Coimbatore':0.9,'Kochi':1.0,'Chandigarh':1.1,
    'Goa':1.3,'Mysore':1.0}

WARNINGS = {
    'Wedding':     {'per_guest':3500,'min':300000},
    'Corporate':   {'per_guest':2500,'min':150000},
    'Birthday':    {'per_guest':1500,'min':50000},
    'Anniversary': {'per_guest':2000,'min':80000},
    'Party':       {'per_guest':1200,'min':40000},
}

SPLITS = {
    'Wedding':     {'Venue':0.30,'Catering':0.25,'Decoration':0.20,'Photography':0.15,'Miscellaneous':0.10},
    'Corporate':   {'Venue':0.40,'Catering':0.30,'AV Equipment':0.15,'Miscellaneous':0.15},
    'Birthday':    {'Venue':0.35,'Catering':0.30,'Decoration':0.20,'Cake':0.10,'Entertainment':0.05},
    'Anniversary': {'Venue':0.50,'Decoration':0.25,'Gifts':0.15,'Photography':0.10},
    'Party':       {'Venue':0.35,'Catering':0.35,'Decoration':0.20,'Entertainment':0.10},
}

CHECKLISTS = {
    'Wedding':['12 months before: Book venue','8 months before: Finalize catering',
        '6 months before: Book photographer','3 months before: Send invitations',
        '1 month before: Confirm all vendors','1 week before: Final headcount'],
    'Corporate':['3 months before: Book venue and AV equipment',
        '2 months before: Finalize agenda and speakers','1 month before: Send invitations',
        '2 weeks before: Confirm catering','1 week before: Technical rehearsal'],
    'Birthday':['1 month before: Book venue','3 weeks before: Send invitations',
        '1 week before: Order cake and decorations','1 day before: Setup venue',
        'Day of: Arrange return gifts'],
    'Anniversary':['2 months before: Book venue','1 month before: Send invitations',
        '1 week before: Arrange flowers and cake','1 day before: Setup decorations',
        'Day of: Coordinate with photographer'],
    'Party':['1 month before: Decide venue and theme','3 weeks before: Send invitations',
        '1 week before: Arrange food and drinks','1 day before: Setup decorations',
        'Day of: Arrange music and entertainment'],
}

KNOWLEDGE_BASE = [
    "A typical wedding timeline starts with venue booking 12 months before, catering 8 months before, photography 6 months before, and decoration 2 months before the wedding date.",
    "Wedding budget in India typically splits as: venue 30%, catering 25%, decoration 20%, photography 15%, and miscellaneous 10%.",
    "For a wedding in Bangalore, popular venues include hotels in Indiranagar, Koramangala and Whitefield. Average cost for 200 guests is between 8 to 15 lakhs.",
    "Wedding checklist must include: venue, catering, decoration, photographer, videographer, mehendi artist, music and DJ, bridal wear, and invitation cards.",
    "Outdoor weddings require backup plans for rain, extra lighting arrangements, and generator backup for power.",
    "A corporate event planning checklist includes: venue selection, audio visual equipment, catering, guest list, agenda planning, and post event feedback.",
    "Corporate event budgets typically allocate 40% for venue, 30% for catering, 15% for AV equipment, and 15% for miscellaneous expenses.",
    "For a successful product launch event, plan 3 months in advance, hire a professional MC, arrange live demonstrations, and prepare press kits.",
    "Team building events work best with outdoor activities, group challenges, and informal networking sessions over meals.",
    "Corporate conferences need reliable WiFi, projectors, microphones, breakout rooms, and catering for tea and lunch breaks.",
    "A birthday party timeline: send invites 3 weeks before, book venue 1 month before, order cake 1 week before, arrange decorations 1 day before.",
    "Kids birthday party essentials include: return gifts, games and activities, themed decoration, birthday cake, and a designated play area.",
    "Birthday party budget breakdown: venue 35%, catering 30%, decoration 20%, cake 10%, and entertainment 5%.",
    "Surprise birthday parties require a trusted coordinator, a distraction plan, and guests who can keep secrets.",
    "Popular birthday party themes in India include Bollywood, superheroes, princesses, outdoor adventure, and retro themes.",
    "Anniversary celebration ideas include: romantic dinner at a rooftop restaurant, destination trip, surprise party with close friends and family.",
    "25th silver anniversary traditionally features silver decorations, a photo slideshow of memories, and renewal of wedding vows.",
    "Anniversary party planning timeline: book venue 2 months before, send invites 1 month before, arrange flowers and cake 1 week before.",
    "For an intimate anniversary dinner, choose a private dining room, arrange flowers and candles, hire a live musician for background music.",
    "Anniversary budget typically splits as: venue and food 50%, decoration 25%, gifts and mementos 15%, photography 10%.",
    "House warming party essentials: invite neighbours and close friends, arrange housewarming gifts display, prepare home cooked food or hire a caterer.",
    "New year party planning: book venue 3 months in advance as December dates fill up fast, arrange countdown timer, DJ, and midnight snacks.",
    "Farewell party checklist includes: personalized gifts, memory book with messages from colleagues, favourite food of the person leaving.",
    "Graduation party ideas include outdoor barbecue, photo booth with props, slideshow of college memories, and a customized graduation cake.",
    "Reunion party planning tips: create a WhatsApp group for coordination, collect contributions digitally, arrange a photographer, and prepare a quiz about shared memories.",
]

# ── MODEL LOADING ─────────────────────────────────────────
@st.cache_resource
def load_models():
    import os
    base = os.path.dirname(os.path.abspath(__file__))
    nlp_model = joblib.load(os.path.join(base, 'models', 'nlp_model.pkl'))
    vectorizer = joblib.load(os.path.join(base, 'models', 'vectorizer.pkl'))
    index = faiss.read_index(os.path.join(base, 'models', 'knowledge_index.faiss'))
    embed_model = SentenceTransformer('all-MiniLM-L6-v2',
                  cache_folder=os.path.join(base, 'models', 'sentence_transformer_cache'))
    return nlp_model, vectorizer, index, embed_model
# ── PLANNING LOGIC ────────────────────────────────────────
def generate_plan(event_type, city, guest_count, duration_days, season, outdoor, user_budget):
    _, _, index, embed_model = load_models()
    city_mult = CITY_MULT.get(city, 1.0)
    min_required = (WARNINGS[event_type]['per_guest'] * guest_count + WARNINGS[event_type]['min']) * city_mult

    warning = None
    if user_budget < min_required * 0.5:
        warning = f"Your budget of ₹{user_budget:,} is very low for {guest_count} guests in {city}. Minimum recommended is ₹{int(min_required):,}. Plan adjusted to minimum viable budget."
        working_budget = int(min_required * 0.5)
    elif user_budget < min_required:
        warning = f"Your budget of ₹{user_budget:,} is a bit tight for {guest_count} guests in {city}. We recommend at least ₹{int(min_required):,}. We've created the best plan within your budget."
        working_budget = user_budget
    else:
        working_budget = user_budget

    breakdown = {k: int(working_budget * v) for k, v in SPLITS[event_type].items()}
    query_vector = embed_model.encode([f"{event_type} planning tips"])
    distances, indices = index.search(np.array(query_vector), 2)
    tips = [KNOWLEDGE_BASE[i] for i in indices[0]]

    return {
        'event_type': event_type,
        'city': city,
        'guest_count': guest_count,
        'duration_days': duration_days,
        'user_budget': user_budget,
        'working_budget': working_budget,
        'warning': warning,
        'budget_breakdown': breakdown,
        'checklist': CHECKLISTS[event_type],
        'tips': tips,
    }

# ── SESSION STATE ─────────────────────────────────────────
if 'selected_event' not in st.session_state:
    st.session_state.selected_event = None
if 'show_results' not in st.session_state:
    st.session_state.show_results = False
if 'result_data' not in st.session_state:
    st.session_state.result_data = None

# ── HERO ──────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-badge">🎉 AI-Powered Event Planning</div>
  <div class="hero-title">Plan your perfect<br/>event with AI</div>
  <div class="hero-sub">Smart budget planning, checklists, and tips — all in seconds.</div>
</div>
""", unsafe_allow_html=True)

# ── INPUT FORM ────────────────────────────────────────────
if not st.session_state.show_results:
    st.markdown('<div class="form-card">', unsafe_allow_html=True)
    st.markdown('<div class="form-section-title">Step 1 — Select your event type</div>', unsafe_allow_html=True)

    events = [("💍","Wedding"),("🏢","Corporate"),("🎂","Birthday"),("💑","Anniversary"),("🎊","Party")]
    cols = st.columns(5)
    for i, (icon, name) in enumerate(events):
        with cols[i]:
            if st.button(f"{icon}\n{name}", key=f"evt_{name}", use_container_width=True):
                st.session_state.selected_event = name
                st.rerun()
            if st.session_state.selected_event == name:
                st.markdown('<div style="height:3px;background:#7c3aed;border-radius:2px;margin-top:-8px"></div>', unsafe_allow_html=True)

    if st.session_state.selected_event:
        st.markdown(f'<div style="color:#a78bfa;font-size:13px;margin:12px 0 20px">✦ Selected: {st.session_state.selected_event}</div>', unsafe_allow_html=True)
        st.markdown('<div class="form-section-title" style="margin-top:24px">Step 2 — Event details</div>', unsafe_allow_html=True)

        with st.form("event_form"):
            preferences = st.text_input("Any preferences? (optional)", placeholder="e.g. garden theme, vegetarian food, live music...")
            col1, col2, col3 = st.columns(3)
            with col1:
                city = st.selectbox("City", CITIES)
                guest_count = st.number_input("Number of guests", min_value=1, max_value=2000, value=100)
            with col2:
                duration_days = st.selectbox("Duration", [1,2,3], format_func=lambda x: f"{x} day{'s' if x>1 else ''}")
                season = st.selectbox("Season", ["Winter","Summer","Monsoon"])
            with col3:
                outdoor = st.selectbox("Venue type", ["Indoors","Outdoors"]) == "Outdoors"
                user_budget = st.number_input("Your budget (₹)", min_value=10000, max_value=100000000, value=500000, step=10000)
            submitted = st.form_submit_button("✦ Generate my event plan")

        if submitted:
            with st.spinner("Our AI agents are planning your event..."):
                result = generate_plan(
                    st.session_state.selected_event, city,
                    guest_count, duration_days, season, outdoor, user_budget
                )
                st.session_state.result_data = result
                st.session_state.show_results = True
                st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

# ── RESULTS ───────────────────────────────────────────────
if st.session_state.show_results and st.session_state.result_data:
    import plotly.graph_objects as go
    d = st.session_state.result_data

    st.markdown('<div class="reset-btn">', unsafe_allow_html=True)
    if st.button("↩ Plan a new event", key="reset"):
        st.session_state.show_results = False
        st.session_state.result_data = None
        st.session_state.selected_event = None
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    if d.get("warning"):
        st.markdown(f'<div class="warning-box">⚠️ {d["warning"]}</div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="result-header">
        <div class="event-type-pill">✦ {d['event_type']}</div>
        <div class="big-budget">₹{d['working_budget']:,}</div>
        <div class="budget-label">Your event budget</div>
        <div class="metric-row">
            <div class="metric-box"><div class="metric-val">{d['guest_count']}</div><div class="metric-lbl">Guests</div></div>
            <div class="metric-box"><div class="metric-val">{d['city']}</div><div class="metric-lbl">City</div></div>
            <div class="metric-box"><div class="metric-val">{d['duration_days']} day{'s' if d['duration_days']>1 else ''}</div><div class="metric-lbl">Duration</div></div>
            <div class="metric-box"><div class="metric-val">₹{d['user_budget']:,}</div><div class="metric-lbl">Your input</div></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.markdown('<div class="section-card"><div class="section-head">Budget Breakdown</div>', unsafe_allow_html=True)
        labels = list(d['budget_breakdown'].keys())
        values = list(d['budget_breakdown'].values())
        colors = ['#7c3aed','#a78bfa','#6d28d9','#4c1d95','#8b5cf6','#c4b5fd']
        fig = go.Figure(data=[go.Pie(
            labels=labels, values=values, hole=0.65,
            marker=dict(colors=colors[:len(labels)], line=dict(color='#0a0a0f', width=2)),
            textinfo='label+percent',
            textfont=dict(color='#e2e8f0', size=12),
        )])
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            showlegend=False, margin=dict(t=0,b=0,l=0,r=0), height=280,
            annotations=[dict(text=f"₹{d['working_budget']:,}", x=0.5, y=0.5,
                font=dict(size=13, color='#a78bfa', family='Inter'), showarrow=False)]
        )
        st.plotly_chart(fig, use_container_width=True, key="budget_chart")
        for cat, amt in d['budget_breakdown'].items():
            st.markdown(f'<div style="display:flex;justify-content:space-between;color:#64748b;font-size:13px;margin-bottom:4px"><span>{cat}</span><span style="color:#e2e8f0">₹{amt:,}</span></div>', unsafe_allow_html=True)
            st.progress(amt / d['working_budget'])
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section-card"><div class="section-head">Event Checklist</div>', unsafe_allow_html=True)
        for i, item in enumerate(d['checklist'], 1):
            st.markdown(f'<div class="check-item"><div class="check-num">{i}</div><div class="check-text">{item}</div></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-card"><div class="section-head">Planning Tips</div>', unsafe_allow_html=True)
        for tip in d['tips']:
            st.markdown(f'<div class="tip-card">{tip}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)