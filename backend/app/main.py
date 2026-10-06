"""WattWise.ai backend — FastAPI application entry point."""

from contextlib import asynccontextmanager
from datetime import datetime
from time import sleep

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api import ai_routes, digital_twin, energy, scenarios_analyst
from app.config import settings
from app.database.connection import Base, engine

seed_result = {"status": "pending", "message": "Database initialization has not run."}


def initialize_database(retries: int = 10, delay_seconds: float = 2.0) -> dict:
    """Create schema and seed data after the database becomes reachable."""
    global seed_result

    from app.database.seed import seed

    last_error = "Database initialization failed."
    for attempt in range(retries):
        try:
            Base.metadata.create_all(bind=engine)
            seed_result = seed()
            return seed_result
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)
            if attempt < retries - 1:
                sleep(delay_seconds)

    seed_result = {"status": "error", "message": f"Seed skipped: {last_error}"}
    return seed_result


@asynccontextmanager
async def lifespan(app: FastAPI):
    initialize_database()
    yield


app = FastAPI(
    title="WattWise.ai API",
    description="AI-powered Smart Energy Optimization & Digital Twin Platform (demo build). "
                "All data is SIMULATED unless explicitly labelled otherwise.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal error: {exc}"},
    )


@app.get("/")
def root():
    return {
        "name": "WattWise.ai API",
        "version": "0.1.0",
        "data_source": settings.DATA_SOURCE_LABEL,
        "seed_status": seed_result.get("status"),
        "seed_message": seed_result.get("message", ""),
        "endpoints": [
            "/api/buildings", "/api/zones", "/api/energy/current",
            "/api/energy/history", "/api/analytics", "/api/forecast",
            "/api/anomalies", "/api/recommendations", "/api/digital-twin",
            "/api/digital-twin/simulate/zone", "/api/digital-twin/simulate/building",
            "/api/digital-twin/optimize", "/api/scenarios", "/api/analyst",
        ],
    }


@app.get("/health")
def health():
    from sqlalchemy import text

    from app.database.connection import SessionLocal

    db_ok = True
    try:
        db = SessionLocal()
        db.execute(text("SELECT 1"))
        db.close()
    except Exception:  # noqa: BLE001
        db_ok = False
    return {
        "status": "ok" if db_ok else "degraded",
        "database": "connected" if db_ok else "unavailable",
        "time": datetime.utcnow().isoformat(),
    }


app.include_router(energy.router, prefix="/api")
app.include_router(ai_routes.router, prefix="/api")
app.include_router(digital_twin.router, prefix="/api")
app.include_router(scenarios_analyst.router, prefix="/api")