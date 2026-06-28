<div align="center">
<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=28&pause=1000&color=7C6AFF&center=true&vCenter=true&width=600&lines=Two-Stage+Recommender+System;Production+ML+Architecture" alt="Typing SVG" />
<br/>
**The recommendation pipeline used by Amazon, YouTube, and Meta — built from scratch.**  
Candidate generation with ALS + FAISS, re-ranked with LightGBM LambdaMART. Served via FastAPI with a live React dashboard.
 
<br/>
[![Python](https://img.shields.io/badge/Python_3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React_18-20232A?style=for-the-badge&logo=react&logoColor=61DAFB)](https://react.dev)
[![LightGBM](https://img.shields.io/badge/LightGBM-F7931E?style=for-the-badge)](https://lightgbm.readthedocs.io)
[![FAISS](https://img.shields.io/badge/FAISS-1877F2?style=for-the-badge)](https://faiss.ai)
 
</div>
---
 
## Overview
 
Most recommender tutorials train one model and call it done. Real systems don't work that way — you can't run a heavy ranking model over millions of items at request time. This project implements the actual production pattern: a two-stage pipeline that **retrieves cheaply, then ranks precisely**.
 
```
500,000 items  ──►  Stage 1: ALS + FAISS  ──►  200 candidates  ──►  Stage 2: LightGBM  ──►  10 results
                      ~2ms · cosine ANN                                ~12ms · LambdaMART
```
 
---
 
## Features
 
- **Stage 1 — Candidate Generation** · Alternating Least Squares on implicit feedback, embedded into a FAISS index for sub-5ms approximate nearest-neighbor retrieval
- **Stage 2 — Ranking** · LightGBM with LambdaMART listwise loss, trained on rich features: embedding similarity, item metadata, user history, and recency
- **Temporal evaluation** · Train/test split by timestamp, not random — mirrors real deployment and avoids future leakage
- **Cold start handling** · Popularity-based fallback for new users with no embedding
- **Live dashboard** · React frontend showing the pipeline trace, per-stage latency, feature importance, and offline metrics
- **REST API** · `POST /recommend` returns ranked results with full pipeline introspection
---
 
## Quickstart
 
```bash
# Clone and enter
git clone https://github.com/you/recsys && cd recommender-system
 
# One-command setup: creates venv, installs deps, generates data, trains both models
chmod +x setup.sh && ./setup.sh
```
 
Open two terminals:
 
```bash
# Terminal 1 — API server
cd backend
source .venv/bin/activate
uvicorn main:app --reload
# Swagger docs → http://localhost:8000/docs
```
 
```bash
# Terminal 2 — React dashboard
cd frontend
npm start
# Dashboard → http://localhost:3000
```
 
---
 
## Manual Setup
 
<details>
<summary><b>Backend</b></summary>
```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
 
# Generate 2,000 users · 500 items · 80,000 interactions
python generate_data.py
 
# Train Stage 1 (ALS + FAISS)
python -c "from stage1_candidate_gen import train_and_save; train_and_save()"
 
# Train Stage 2 (LightGBM LambdaMART)
python -c "from stage2_ranker import train_and_save; train_and_save()"
 
# Start server
uvicorn main:app --reload --port 8000
```
 
</details>
<details>
<summary><b>Frontend</b></summary>
```bash
cd frontend
npm install
npm start
```
 
</details>
---
 
## Project Structure
 
```
recommender-system/
├── backend/
│   ├── main.py                  # FastAPI app — /recommend, /metrics, /user, /health
│   ├── generate_data.py         # Synthetic MovieLens-style data generator
│   ├── stage1_candidate_gen.py  # ALS training + FAISS index + Recall@K eval
│   ├── stage2_ranker.py         # LightGBM LambdaMART + feature engineering + NDCG eval
│   ├── train.py                 # End-to-end training orchestrator
│   └── requirements.txt
│
├── frontend/
│   └── src/App.js               # Dashboard: pipeline trace, metrics, architecture view
│
├── data/                        # Generated .parquet files  [gitignored]
├── models/                      # Trained model artifacts   [gitignored]
├── setup.sh                     # One-command bootstrap
└── recommender.code-workspace   # VS Code multi-root workspace
```
 
---
 
## API Reference
 
### `POST /recommend`
 
```json
{
  "user_id": 42,
  "n_candidates": 200,
  "n_results": 10
}
```
 
```jsonc
// Response
{
  "user_id": 42,
  "total_latency_ms": 14.3,
  "cold_start": false,
  "results": [
    {
      "item_id": 137,
      "title": "Blade Runner 2049",
      "genres": ["Sci-Fi", "Drama"],
      "rank": 1,
      "stage1_score": 0.8821,
      "stage2_score": 3.4102
    }
  ],
  "pipeline": [
    { "name": "Stage 1 — Candidate Generation", "latency_ms": 2.1, "n_items_out": 200 },
    { "name": "Stage 2 — Ranking",              "latency_ms": 11.4, "n_items_out": 10 }
  ]
}
```
 
| Endpoint | Description |
|---|---|
| `POST /recommend` | Run the full two-stage pipeline for a user |
| `GET /user/{id}` | User profile, top genres, interaction history |
| `GET /metrics` | Live latency percentiles + offline eval metrics |
| `GET /health` | Model and data load status |
| `GET /docs` | Interactive Swagger UI |
 
---
 
## Evaluation
 
Metrics are reported per-stage so you can debug the funnel independently.
 
| Stage | Metric | What it measures |
|---|---|---|
| Stage 1 | **Recall@200** | Does the right item make it into candidates? |
| Stage 2 | **NDCG@10** | Is the best item near the top of the ranked list? |
| Stage 2 | **MAP** | Average precision across all users |
| Live | **p50 / p95 / p99** | End-to-end latency percentiles |
 
> If NDCG@10 is low but Recall@200 is high, Stage 2 is the bottleneck. If Recall@200 is low, no amount of ranking improvement helps.
 
---
 
## Architecture Decisions
 
**Why two stages?**  
Running a heavy model over 500k items per request costs hundreds of milliseconds. Stage 1 is a cheap embedding lookup and dot-product ANN search (~2ms). Stage 2 applies expensive features to just 200 items (~12ms). The total pipeline is 14ms.
 
**Why LambdaMART over classification?**  
Binary click/no-click framing ignores relative item order. LambdaMART computes pairwise gradients directly from NDCG — the metric that actually matters for ranking quality.
 
**Why temporal splitting?**  
Random splits let future interactions leak into training, inflating offline metrics. Temporal splitting mirrors real deployment: train on the past, evaluate on the future.
 
---
 
## Extending This
 
| Idea | Where to look |
|---|---|
| Replace ALS with a Two-Tower neural model | `stage1_candidate_gen.py` — keep the FAISS index, swap the embedding source |
| Scale to billions of items | Replace `IndexFlatIP` with `IndexIVFFlat` or Google ScaNN |
| Add diversity re-ranking | Post-process Stage 2 with Maximal Marginal Relevance (MMR) |
| Use a real dataset | Drop-in replace `generate_data.py` with MovieLens 25M or H&M Kaggle |
 
---
 
## Tech Stack
 
| Purpose | Library |
|---|---|
| Matrix Factorization | `implicit` — ALS on implicit feedback |
| ANN Search | `faiss-cpu` — IndexFlatIP |
| Ranking | `lightgbm` — LambdaMART objective |
| API | `fastapi` + `uvicorn` |
| Frontend | React 18 + Recharts |
| Data | `pandas` + `pyarrow` |
 
---
 
<div align="center">
**Built to mirror what ships in production, not what gets written in tutorials.**
 
</div>
