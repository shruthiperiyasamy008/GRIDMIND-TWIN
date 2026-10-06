# WattWise.ai

WattWise.ai is a working, hackathon-ready energy optimisation platform with a computational Digital Twin. Every dashboard value in this repository is derived from **SIMULATED** building readings; no live meter or IoT feed is claimed.

## Architecture

`Next.js dashboard → FastAPI API → PostgreSQL → pandas / scikit-learn analytics → Python Digital Twin`

The backend seeds a 30-day hourly simulation for WattWise Demo Campus. `SimulatedDataProvider` is the present provider abstraction; SmartMeter, IoT, and BMS providers are intentionally future extension points.

## Run with Docker

1. Copy `.env.example` to `.env` and set a non-default password.
2. Run `docker compose up --build`.
3. Open http://localhost:3000. The API documentation is at http://localhost:8000/docs.

## Run locally

Start PostgreSQL with the variables in `.env`, then:

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload
```

In another terminal:

```powershell
cd frontend
npm install
npm run dev
```

The backend creates tables and seeds the simulated campus automatically on its first successful database connection. If an existing simulated dataset is more than a day behind, startup shifts its reading timestamps into the current demo period without deleting readings or saved scenarios.

## API

`GET /api/buildings`, `/zones`, `/energy/current`, `/energy/history`, `/analytics`, `/forecast`, `/anomalies`, `/recommendations`, `/digital-twin`, `/scenarios`.

`POST /api/digital-twin/simulate` (or `/simulate/zone`) accepts `{ "zone": "Block A Classroom 1", "hvac": 70, "lighting": 80, "occupancy": 35 }` and recomputes a zone-day energy simulation. `POST /api/digital-twin/optimize` performs deterministic HVAC × lighting scenario search. `POST /api/analyst` accepts `{ "question": "What is causing the peak?" }`.

## ML and twin model

Forecasting uses a scikit-learn RandomForestRegressor on time, temperature, occupancy, lag, and rolling-average features. IsolationForest detects abnormal zone readings. The Digital Twin models zone equipment, occupancy, ambient temperature, schedules, HVAC, lighting, pumps, and other loads across 24 hours. Savings and carbon are explicitly labelled **ESTIMATED**; control outputs are **SIMULATED SCENARIO**.

## Demo route

Open Dashboard for simulated KPIs and detected anomalies, Forecast for the next 24-hour prediction, then Digital Twin. Select Block A, set HVAC from 100% to 70%, run the simulation, and inspect dynamically calculated baseline energy, simulated energy, savings, cost, and peak response. Recommendations and AI Analyst then consume the same analytics and optimisation context.
