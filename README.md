``
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║        ██████╗ ███████╗ ██████╗███████╗██╗   ██╗███████╗    ║
║        ██╔══██╗██╔════╝██╔════╝██╔════╝╚██╗ ██╔╝██╔════╝    ║
║        ██████╔╝█████╗  ██║     ███████╗ ╚████╔╝ ███████╗    ║
║        ██╔══██╗██╔══╝  ██║     ╚════██║  ╚██╔╝  ╚════██║    ║
║        ██║  ██║███████╗╚██████╗███████║   ██║   ███████║    ║
║        ╚═╝  ╚═╝╚══════╝ ╚═════╝╚══════╝   ╚═╝   ╚══════╝    ║
║                                                              ║
║          two-stage recommender  ·  production architecture   ║
╚══════════════════════════════════════════════════════════════╝
```
 
[![Python](https://img.shields.io/badge/python-3.10+-6e56cf?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-00c896?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18-61dafb?style=flat-square&logo=react&logoColor=black)](https://react.dev)
[![LightGBM](https://img.shields.io/badge/LightGBM-LambdaMART-f7931e?style=flat-square)](https://lightgbm.readthedocs.io)
[![FAISS](https://img.shields.io/badge/FAISS-ANN_Search-1877f2?style=flat-square)](https://faiss.ai)
 
</div>
---
 
## the problem
 
Every major recommender system (YouTube, Amazon, Netflix) runs the same architecture. Nowhere online shows you the complete, wired-up version. This is it.
 
```
user_id ──►  500,000 items  →  200 candidates  →  10 results  ──►  response
                  ↑                   ↑                 ↑
           your catalog         Stage 1              Stage 2
                              ALS + FAISS          LightGBM
                              ~2ms lookup          ~12ms rank
```
 
You can't run a heavy model over millions of items in real time. So you don't. You retrieve cheaply, then rank precisely. That's the whole insight.
 
---
 
## the pipeline
 
```
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 1 — CANDIDATE GENERATION                                 │
│                                                                 │
│  algorithm    Alternating Least Squares (ALS)                   │
│  loss         implicit confidence  c = 1 + α·r                 │
│  output       64-dimensional user + item embeddings             │
│  retrieval    FAISS IndexFlatIP  (inner product ANN)            │
│  result       top-200 candidates per user                       │
│  eval         Recall@200                                        │
└────────────────────────────┬────────────────────────────────────┘
                             │ 200 candidates
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 2 — RANKING                                              │
│                                                                 │
│  algorithm    LightGBM with LambdaMART objective                │
│  loss         listwise NDCG optimization                        │
│  features     stage1_score · popularity · avg_rating ·          │
│               user_history · embedding norms · recency          │
│  split        temporal  (train on past, eval on future)         │
│  eval         NDCG@10 · MAP                                     │
└────────────────────────────┬────────────────────────────────────┘
                             │ top-10 ranked
                             ▼
                    POST /recommend
```
 
---
 
## quick start
 
```bash
# clone & enter
git clone https://github.com/you/recsys && cd recsys
 
# one-time setup (creates venv, installs deps, trains both models)
chmod +x setup.sh && ./setup.sh
```
 
Then open two terminals:
 
```bash
# terminal 1 — api
cd backend && source .venv/bin/activate
uvicorn main:app --reload
# → http://localhost:8000/docs
 
# terminal 2 — ui
cd frontend && npm start
# → http://localhost:3000
```
 
---
 
## manual setup
 
```bash
# backend
cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
 
# generate 2k users · 500 items · 80k interactions
python generate_data.py
 
# train stage 1  (ALS + FAISS index)
python -c "from stage1_candidate_gen import train_and_save; train_and_save()"
 
# train stage 2  (LightGBM LambdaMART)
python -c "from stage2_ranker import train_and_save; train_and_save()"
 
# serve
uvicorn main:app --reload
```
 
```bash
# frontend
cd frontend && npm install && npm start
```
 
---
 
## project layout
 
```
recommender-system/
│
├── backend/
│   ├── main.py                  FastAPI server · /recommend · /metrics · /user
│   ├── generate_data.py         synthetic data generator (MovieLens-style)
│   ├── stage1_candidate_gen.py  ALS training · FAISS index build · Recall@K eval
│   ├── stage2_ranker.py         LightGBM LambdaMART · feature engineering · NDCG eval
│   ├── train.py                 end-to-end training orchestrator
│   └── requirements.txt
│
├── frontend/
│   └── src/App.js               React dashboard · pipeline trace · metrics · arch view
│
├── data/                        generated .parquet files  (git-ignored)
├── models/                      trained artifacts  (git-ignored)
├── setup.sh                     one-command bootstrap
└── recommender.code-workspace   VS Code multi-root workspace
```
 
---
 
## api
 
```http
POST /recommend
Content-Type: application/json
 
{
  "user_id": 42,
  "n_candidates": 200,
  "n_results": 10
}
```
 
```jsonc
// response
{
  "user_id": 42,
  "results": [
    {
      "item_id": 137,
      "title": "Blade Runner 2049",
      "genres": ["Sci-Fi", "Drama"],
      "rank": 1,
      "stage1_score": 0.8821,   // cosine similarity from ALS embedding
      "stage2_score": 3.4102    // LambdaMART output
    }
    // ...
  ],
  "pipeline": [
    { "name": "Stage 1 — Candidate Generation", "latency_ms": 2.1, "n_items_out": 200 },
    { "name": "Stage 2 — Neural Ranking",        "latency_ms": 11.4, "n_items_out": 10 }
  ],
  "total_latency_ms": 14.3,
  "cold_start": false
}
```
 
Other endpoints:
 
| Endpoint | Returns |
|---|---|
| `GET /user/{id}` | profile · top genres · interaction history · embedding status |
| `GET /metrics` | live p50/p95/p99 latency · Recall@200 · NDCG@10 · feature importance |
| `GET /health` | model load status |
| `GET /docs` | interactive Swagger UI |
 
---
 
## design decisions
 
**why two stages at all?**  
A single heavy model over 500k items at request time costs hundreds of milliseconds. Stage 1 is a cheap dot-product lookup (2ms). Stage 2 runs the expensive model on just 200 items (12ms). This is the architecture YouTube described in their 2016 paper and every major company runs variants of today.
 
**why LambdaMART over classification?**  
Framing ranking as binary click/no-click classification ignores the relative order of items. LambdaMART computes gradients directly from NDCG, which is the metric that actually captures whether the best item is ranked first. Using a classification loss and then re-ranking is two steps in the wrong direction.
 
**why temporal splitting?**  
Random train/test splits allow future interactions to leak into training, producing metrics that won't survive contact with real deployment. Temporal splitting mirrors production: the model only ever sees the past, and is tested on the future.
 
**cold start**  
Users without embeddings (no history, or too recent to appear in training) fall back to popularity-based candidates. Item frequencies follow a power law — the most popular items are the safest bet for a new user. In production you'd combine this with content-based features from onboarding signals.
 
---
 
## evaluation
 
The system reports metrics at two levels, separately for each stage:
 
```
Stage 1  ·  Recall@200      "did the right item make it into candidates?"
Stage 2  ·  NDCG@10         "is the best item near the top of the ranked list?"
         ·  MAP              "across all users, how good is the average precision?"
 
Live     ·  p50 / p95 / p99  latency percentiles across all served requests
```
 
Reporting them separately lets you debug the funnel. If NDCG@10 is low but Recall@200 is high, Stage 2 is the problem. If Recall@200 is low, no amount of ranking improvement will help — the right items aren't even making it to Stage 2.
 
---
 
## extending this
 
| idea | where to touch |
|---|---|
| Replace ALS with a Two-Tower neural model | `stage1_candidate_gen.py` — swap `AlternatingLeastSquares` for a dual-encoder, keep the FAISS index |
| Scale ANN to billions of items | swap `faiss.IndexFlatIP` for `faiss.IndexIVFFlat` or ScaNN |
| Add diversity re-ranking | post-process Stage 2 output with MMR (Maximal Marginal Relevance) |
| Real dataset | drop-in replace `generate_data.py` with MovieLens 25M or the H&M Kaggle dataset |
| Better cold start | add content-based features (genres, year, description embeddings) for items without interaction history |
 
---
 
## stack
 
| | |
|---|---|
| Candidate generation | `implicit` — ALS on implicit feedback |
| ANN retrieval | `faiss-cpu` — IndexFlatIP |
| Ranking | `lightgbm` — LambdaMART objective |
| API | `fastapi` + `uvicorn` |
| Frontend | React 18 + Recharts |
| Data | `pandas` + `pyarrow` parquet |
 
---
 
<div align="center">
<sub>built to mirror what Amazon, YouTube, and Meta actually ship  ·  not a tutorial, a blueprint</sub>
</div>
 
