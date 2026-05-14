import streamlit as st
import requests
import plotly.graph_objects as go

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
.event-btn { background: #111118; border: 1px solid #1e1e2e; border-radius: 14px; padding: 16px; text-align: center; cursor: pointer; transition: all 0.2s; }
.event-btn:hover { border-color: #7c3aed; background: #1a1a2e; }
.event-btn.selected { border-color: #7c3aed; background: #1a1a2e; box-shadow: 0 0 0 1px #7c3aed; }
.event-icon { font-size: 28px; margin-bottom: 8px; }
.event-label { color: #e2e8f0; font-size: 13px; font-weight: 500; }
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
.vendor-card { background: #0a0a0f; border: 1px solid #1e1e2e; border-radius: 12px; padding: 14px 16px; margin-bottom: 10px; display: flex; justify-content: space-between; align-items: center; }
.vendor-name { color: #e2e8f0; font-size: 14px; display: flex; align-items: center; gap: 8px; }
.vendor-dot { width: 6px; height: 6px; border-radius: 50%; background: #7c3aed; }
.vendor-cost { color: #a78bfa; font-size: 13px; font-weight: 600; }
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

# Session state for selected event and results
if 'selected_event' not in st.session_state:
    st.session_state.selected_event = None
if 'show_results' not in st.session_state:
    st.session_state.show_results = False
if 'result_data' not in st.session_state:
    st.session_state.result_data = None

# Hero
st.markdown("""
<div class="hero">
  <div class="hero-badge">🎉 AI-Powered Event Planning</div>
  <div class="hero-title">Plan your perfect<br/>event with AI</div>
  <div class="hero-sub">Smart budget planning, checklists, and vendor recommendations — all in seconds.</div>
</div>
""", unsafe_allow_html=True)

# Event type selector
if not st.session_state.show_results:
    st.markdown('<div class="form-card">', unsafe_allow_html=True)
    st.markdown('<div class="form-section-title">Step 1 — Select your event type</div>', unsafe_allow_html=True)

    events = [
        ("💍", "Wedding"),
        ("🏢", "Corporate"),
        ("🎂", "Birthday"),
        ("💑", "Anniversary"),
        ("🎊", "Party"),
    ]

    cols = st.columns(5)
    for i, (icon, name) in enumerate(events):
        with cols[i]:
            selected = st.session_state.selected_event == name
            if st.button(f"{icon}\n{name}", key=f"evt_{name}", use_container_width=True):
                st.session_state.selected_event = name
                st.rerun()
            if selected:
                st.markdown(f'<div style="height:3px;background:#7c3aed;border-radius:2px;margin-top:-8px"></div>', unsafe_allow_html=True)

    if st.session_state.selected_event:
        st.markdown(f'<div style="color:#a78bfa;font-size:13px;margin:12px 0 20px">✦ Selected: {st.session_state.selected_event}</div>', unsafe_allow_html=True)

        st.markdown('<div class="form-section-title" style="margin-top:24px">Step 2 — Event details</div>', unsafe_allow_html=True)

        # Fetch cities from backend
        try:
            cities_resp = requests.get("http://127.0.0.1:8000/cities")
            cities = cities_resp.json()["cities"]
        except:
            cities = ["Bangalore", "Mumbai", "Delhi", "Chennai", "Hyderabad"]

        with st.form("event_form"):
            preferences = st.text_input("Any preferences? (optional)", placeholder="e.g. outdoor garden theme, vegetarian food, live music...")
            col1, col2, col3 = st.columns(3)
            with col1:
                city = st.selectbox("City", cities)
                guest_count = st.number_input("Number of guests", min_value=1, max_value=2000, value=100)
            with col2:
                duration_days = st.selectbox("Duration", [1, 2, 3], format_func=lambda x: f"{x} day{'s' if x > 1 else ''}")
                season = st.selectbox("Season", ["Winter", "Summer", "Monsoon"])
            with col3:
                outdoor = st.selectbox("Venue type", ["Indoors", "Outdoors"]) == "Outdoors"
                user_budget = st.number_input("Your budget (₹)", min_value=10000, max_value=100000000, value=500000, step=10000)

            submitted = st.form_submit_button("✦ Generate my event plan")

        if submitted:
            with st.spinner("Our AI agents are planning your event..."):
                try:
                    response = requests.post("http://127.0.0.1:8000/plan", json={
                        "event_type": st.session_state.selected_event,
                        "preferences": preferences,
                        "city": city,
                        "guest_count": guest_count,
                        "duration_days": duration_days,
                        "season": season,
                        "outdoor": outdoor,
                        "user_budget": user_budget
                    })
                    st.session_state.result_data = response.json()
                    st.session_state.show_results = True
                    st.rerun()
                except Exception as e:
                    st.error(f"Make sure FastAPI backend is running! Error: {str(e)}")

    st.markdown('</div>', unsafe_allow_html=True)

# Results page
if st.session_state.show_results and st.session_state.result_data:
    d = st.session_state.result_data

    # New plan button
    st.markdown('<div class="reset-btn">', unsafe_allow_html=True)
    if st.button("↩ Plan a new event", key="reset"):
        st.session_state.show_results = False
        st.session_state.result_data = None
        st.session_state.selected_event = None
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    # Warning box
    if d.get("warning"):
        st.markdown(f'<div class="warning-box">⚠️ {d["warning"]}</div>', unsafe_allow_html=True)

    # Result header
    st.markdown(f"""
    <div class="result-header">
        <div class="event-type-pill">✦ {d['event_type']}</div>
        <div class="big-budget">₹{d['working_budget']:,}</div>
        <div class="budget-label">Your event budget</div>
        <div class="metric-row">
            <div class="metric-box"><div class="metric-val">{d['guest_count']}</div><div class="metric-lbl">Guests</div></div>
            <div class="metric-box"><div class="metric-val">{d['city']}</div><div class="metric-lbl">City</div></div>
            <div class="metric-box"><div class="metric-val">{d['duration_days']} day{'s' if d['duration_days'] > 1 else ''}</div><div class="metric-lbl">Duration</div></div>
            <div class="metric-box"><div class="metric-val">₹{d['user_budget']:,}</div><div class="metric-lbl">Your input</div></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1.2, 1])

    with col1:
        # Budget donut chart
        st.markdown('<div class="section-card"><div class="section-head">Budget Breakdown</div>', unsafe_allow_html=True)
        labels = list(d['budget_breakdown'].keys())
        values = list(d['budget_breakdown'].values())
        colors = ['#7c3aed', '#a78bfa', '#6d28d9', '#4c1d95', '#8b5cf6', '#c4b5fd']
        fig = go.Figure(data=[go.Pie(
            labels=labels, values=values,
            hole=0.65,
            marker=dict(colors=colors[:len(labels)], line=dict(color='#0a0a0f', width=2)),
            textinfo='label+percent',
            textfont=dict(color='#e2e8f0', size=12),
        )])
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            showlegend=False,
            margin=dict(t=0, b=0, l=0, r=0),
            height=280,
            annotations=[dict(
                text=f"₹{d['working_budget']:,}",
                x=0.5, y=0.5,
                font=dict(size=13, color='#a78bfa', family='Inter'),
                showarrow=False
            )]
        )
        st.plotly_chart(fig, use_container_width=True, key="budget_chart")

        for cat, amt in d['budget_breakdown'].items():
            pct = amt / d['working_budget']
            st.markdown(f'<div style="display:flex;justify-content:space-between;color:#64748b;font-size:13px;margin-bottom:4px"><span>{cat}</span><span style="color:#e2e8f0">₹{amt:,}</span></div>', unsafe_allow_html=True)
            st.progress(pct)
        st.markdown('</div>', unsafe_allow_html=True)

        
    with col2:
        # Checklist
        st.markdown('<div class="section-card"><div class="section-head">Event Checklist</div>', unsafe_allow_html=True)
        for i, item in enumerate(d['checklist'], 1):
            st.markdown(f"""
            <div class="check-item">
                <div class="check-num">{i}</div>
                <div class="check-text">{item}</div>
            </div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Tips
        st.markdown('<div class="section-card"><div class="section-head">Planning Tips</div>', unsafe_allow_html=True)
        for tip in d['tips']:
            st.markdown(f'<div class="tip-card">{tip}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)