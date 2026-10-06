from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.config import settings
from app.database.connection import get_db
from app.database.models import Recommendation, Scenario
from app.schemas import AnalystRequest, ScenarioCreate

router = APIRouter()


@router.get("/scenarios")
def list_scenarios(db: Session = Depends(get_db)):
    rows = db.query(Scenario).order_by(Scenario.created_at.desc()).all()
    return {
        "source": settings.DATA_SOURCE_LABEL,
        "scenarios": [
            {
                "id": s.id, "name": s.name, "created_at": s.created_at.isoformat(),
                "hvac_percentage": s.hvac_percentage,
                "lighting_percentage": s.lighting_percentage,
                "occupancy_percentage": s.occupancy_percentage,
                "baseline_energy": s.baseline_energy,
                "simulated_energy": s.simulated_energy,
                "estimated_savings": s.estimated_savings,
                "estimated_cost_savings": s.estimated_cost_savings,
                "labels": {"energy": "ESTIMATED", "scenario": "SIMULATED SCENARIO"},
            }
            for s in rows
        ],
    }


@router.post("/scenarios")
def create_scenario(req: ScenarioCreate, db: Session = Depends(get_db)):
    s = Scenario(
        name=req.name,
        hvac_percentage=req.hvac_percentage,
        lighting_percentage=req.lighting_percentage,
        occupancy_percentage=req.occupancy_percentage,
        baseline_energy=req.baseline_energy,
        simulated_energy=req.simulated_energy,
        estimated_savings=req.estimated_savings,
        estimated_cost_savings=req.estimated_cost_savings,
    )
    db.add(s)
    db.commit()
    db.refresh(s)
    return {"status": "created", "id": s.id, "name": s.name}


@router.get("/recommendations/dismiss/{rec_id}")
def dismiss_recommendation(rec_id: int, db: Session = Depends(get_db)):
    rec = db.query(Recommendation).get(rec_id)
    if not rec:
        raise HTTPException(status_code=404, detail="Recommendation not found.")
    rec.status = "DISMISSED"
    db.commit()
    return {"status": "dismissed", "id": rec_id}


@router.post("/analyst")
def analyst(req: AnalystRequest, db: Session = Depends(get_db)):
    from datetime import timedelta

    from app.ai.analyst import answer
    from app.ai.anomaly_detection import AnomalyDetector
    from app.ai.optimization import optimize_twin
    from app.services import analytics_service
    from app.digital_twin.twin_service import build_twin_from_db

    df = analytics_service.building_level_frame(db, start=None)
    if df.empty:
        raise HTTPException(status_code=503, detail="No energy data available for analysis.")

    zone_totals = analytics_service.zone_totals(db)
    zone_names = {z.id: z.name for z in analytics_service.zone_map(db).values()}
    detections = AnomalyDetector().detect(
        analytics_service.readings_frame(db, start=None), zone_names=zone_names)[:5]

    twin, zone_energy_map = build_twin_from_db(db)
    optimization = optimize_twin(twin, zone_energy_map)

    zones_meta = [{"id": z.id, "name": z.name, "zone_type": z.zone_type}
                  for z in analytics_service.zone_map(db).values()]

    return answer(req.question, df, zone_totals, detections, optimization, zones_meta)