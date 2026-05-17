# 🎉 Festiva Planner AI

> An AI-powered event planning assistant that generates complete event plans, predicts budgets using ML, retrieves knowledge using RAG, and coordinates multiple agents — all from a single user input.

**Live Demo:** [festiva-planner-production.up.railway.app](https://festiva-planner-production.up.railway.app)

---

## What it does

Type something like _"I want to plan a wedding in Bangalore for 200 guests"_ and Festiva will instantly generate:

- ✅ A complete event timeline and checklist
- ✅ A smart budget prediction based on city, guests, and event type
- ✅ A detailed budget breakdown with an interactive donut chart
- ✅ Planning tips retrieved from a knowledge base using RAG
- ✅ Budget warnings if your input is too low for your requirements
- ✅ An AI assistant to answer follow-up questions
- ✅ A downloadable PDF of your complete event plan

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit |
| ML Model | Scikit-learn (Random Forest Regressor) |
| NLP Classifier | TF-IDF + Logistic Regression |
| Knowledge Retrieval | FAISS + Sentence Transformers (RAG) + Out-of-scope Detection |
| Agent Orchestration | LangChain-style multi-agent pipeline |
| PDF Export | ReportLab |
| Deployment | Railway + Docker |

---

## Architecture

```
User Input
    │
    ▼
┌─────────────────────────────────────┐
│           Streamlit UI              │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│         Multi-Agent System          │
│                                     │
│  ┌───────────┐  ┌───────────────┐   │
│  │  Planner  │  │    Budget     │   │
│  │   Agent   │  │    Agent      │   │
│  │  (NLP +   │  │  (ML Model)   │   │
│  │ Checklist)│  │               │   │
│  └───────────┘  └───────────────┘   │
│                                     │
│  ┌───────────────────────────────┐  │
│  │      Knowledge Agent          │  │
│  │   (FAISS + RAG Retrieval)     │  │
│  └───────────────────────────────┘  │
└─────────────────────────────────────┘
    │
    ▼
Complete Event Plan + Budget + Tips + PDF
```

---

## Project Structure

```
festiva-planner/
├── app.py                  ← Main Streamlit app (all agents + UI)
├── Dockerfile              ← Docker config for deployment
├── requirements.txt        ← Python dependencies
├── data/
│   └── events.csv          ← Synthetic event dataset (200 rows)
├── models/
│   ├── budget_model.pkl    ← Trained Random Forest model
│   ├── nlp_model.pkl       ← Trained NLP classifier
│   ├── vectorizer.pkl      ← TF-IDF vectorizer
│   ├── knowledge_base.pkl  ← Event planning knowledge base
│   └── knowledge_index.faiss ← FAISS vector index
└── notebooks/
    ├── week1_eda.ipynb         ← Data exploration
    ├── week2_budget_model.ipynb ← ML model training
    ├── week3_nlp.ipynb         ← NLP classifier training
    ├── week4_rag.ipynb         ← RAG knowledge base setup
    └── week5_agents.ipynb      ← Multi-agent system
```

---

## 6-Week Development Plan

### Week 1 — Data & Problem Setup
- Defined 5 event types: Wedding, Corporate, Birthday, Anniversary, Party
- Generated a synthetic dataset of 200 events with realistic Indian pricing
- Performed EDA to understand budget patterns across event types and cities

### Week 2 — ML Budget Prediction Model
- Built a Random Forest Regressor to predict event budgets
- Features: event type, city, guest count, duration, season, venue type
- Achieved **R2 score of 0.85** (85% accuracy)
- Key insight: data quality matters more than model complexity

### Week 3 — NLP Event Classifier
- Built a TF-IDF + Logistic Regression classifier
- Classifies user input into one of 5 event types from plain English
- Achieved **80% accuracy** on test set
- Example: _"surprise party for my best friend"_ → correctly classified as Birthday

### Week 4 — RAG Knowledge Base
- Created a knowledge base of 25 event planning guides
- Used `sentence-transformers` (all-MiniLM-L6-v2) to convert text to vectors
- Stored vectors in FAISS for fast similarity search
- Retrieves the most relevant planning tips for any user question

### Week 5 — Multi-Agent System
- **Planner Agent:** detects event type using NLP, generates checklist
- **Budget Agent:** predicts budget using ML model, splits into categories
- **Knowledge Agent:** searches FAISS knowledge base using RAG
- All three agents work together from a single user input

### Week 6 — Deployment
- Merged all agents into a single Streamlit app
- Added smart budget warnings based on real Indian market rates
- Added AI assistant chat using RAG
- Added PDF export using ReportLab
- Deployed to Railway with Docker for a single public link

---

## How to Run Locally

**1. Clone the repository**
```bash
git clone https://github.com/ShreyasJRao/festiva-planner.git
cd festiva-planner
```

**2. Create and activate virtual environment**
```bash
python -m venv venv
venv\Scripts\activate   # Windows
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Run the app**
```bash
streamlit run app.py
```

**5. Open in browser**
```
http://localhost:8501
```

---

## Key Features

### Smart Budget Warnings
The system compares your budget against real Indian market rates per guest per city. If your budget is too low, it warns you with a specific recommendation instead of generating an unrealistic plan.

### City-aware Pricing
Supports 24 major Indian cities with realistic cost multipliers. Mumbai and Delhi are 40-50% more expensive than tier-2 cities like Patna or Kanpur.

### RAG-powered AI Assistant
After generating your plan, you can ask follow-up questions about dress codes, food, decorations, photography, venues, and more. The assistant uses two layers of intelligence:
- **Out-of-scope detection** — if a question is unrelated to event planning, it politely redirects instead of returning irrelevant results
- **Extended knowledge base** — covers dress codes, food menus, decoration ideas, photography tips, venue selection, and music recommendations
- **Dual search** — searches both the original FAISS index and an extended knowledge base, returning the 3 most relevant answers

---

## Sample Output

**Input:** Wedding in Bangalore, 200 guests, ₹10,00,000 budget, 2 days, Winter, Indoors

**Output:**
- Event Type: Wedding (detected automatically by NLP)
- Total Budget: ₹10,00,000
- Venue: ₹3,00,000 | Catering: ₹2,50,000 | Decoration: ₹2,00,000 | Photography: ₹1,50,000 | Misc: ₹1,00,000
- 6-step checklist from 12 months to 1 week before
- 2 relevant planning tips from knowledge base
- Downloadable PDF plan

---

## Built By

**Shreyas J Rao** — Internship Capstone Project

> Built from scratch over 6 weeks using Python, Scikit-learn, FAISS, Sentence Transformers, Streamlit, and Railway.
