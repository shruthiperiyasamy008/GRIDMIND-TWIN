from datetime import timedelta

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.ai import forecasting
from app.ai.analyst import answer as analyst_answer
from app.ai.anomaly_detection import AnomalyDetector, peak_demand_warning
from app.ai.optimization import optimize_twin
from app.ai.recommendations import generate_recommendations
from app.config import settings
from app.database.connection import get_db
from app.services import analytics_service

router = APIRouter()


@router.get("/forecast")
def forecast(db: Session = Depends(get_db)):
    try:
        df = analytics_service.building_level_frame(db, start=None)
        result = forecasting.forecast_service.get_forecast(df)
        return {
            **result,
            "labels": {"type": "PREDICTION"},
        }
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Forecast failure: {exc}")


@router.get("/anomalies")
def anomalies(db: Session = Depends(get_db)):
    df = analytics_service.readings_frame(db, start=None)
    if df.empty:
        raise HTTPException(status_code=503, detail="No data available for anomaly detection.")
    zone_names = {z.id: z.name for z in analytics_service.zone_map(db).values()}
    detector = AnomalyDetector()
    detections = detector.detect(df, zone_names=zone_names)
    df_b = analytics_service.building_level_frame(db, start=None)
    peak_warns = peak_demand_warning(df_b) if not df_b.empty else []

    # cluster detections with associated zone meta for the frontend
    for d in detections:
        d["severity_rank"] = {"CRITICAL": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1}.get(d["severity"], 0)

    return {
        "source": settings.DATA_SOURCE_LABEL,
        "detections": detections,
        "peak_warnings": peak_warns,
        "count": len(detections),
    }


@router.get("/recommendations")
def recommendations(db: Session = Depends(get_db)):
    try:
        df = analytics_service.building_level_frame(db, start=None)
        zone_totals = analytics_service.zone_totals(db)
        detections = []
        if not df.empty:
            zone_names = {z.id: z.name for z in analytics_service.zone_map(db).values()}
            detections = AnomalyDetector().detect(
                analytics_service.readings_frame(db, start=None), zone_names=zone_names)[:5]
        opt = optimize_twin_ctx(db, df)
        recs = generate_recommendations(db, detections, zone_totals, opt)
        return {"source": settings.DATA_SOURCE_LABEL, "recommendations": recs,
                "labels": {"savings": "ESTIMATED"}}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Recommendation failure: {exc}")


def optimize_twin_ctx(db: Session, df=None):
    from app.digital_twin.twin_service import build_twin_from_db
    from app.services.analytics_service import zone_totals

    twin, zone_energy_map = build_twin_from_db(db)
    if df is not None and not df.empty:
        pass
    return optimize_twin(twin, zone_energy_map)


@router.get("/analytics")
def analytics(
    granularity: str = "daily",
    building_id: int | None = None,
    zone_id: int | None = None,
    db: Session = Depends(get_db),
):
    from dateutil.relativedelta import relativedelta
    from datetime import datetime

    end = datetime.now()
    start = end - relativedelta(days=30)
    df = analytics_service.building_level_frame(db, start=start, end=end,
                                                building_id=building_id)
    if df.empty:
        raise HTTPException(status_code=503, detail="No energy data available.")

    series = analytics_service.readings_frame(db, start=start, end=end, zone_id=zone_id)
    zone_totals = analytics_service.zone_totals(db, start=start, end=end)
    eq_split = energy_equipment(df)

    summary = energy_summary(df)

    from app.ai.forecasting import forecast_service
    try:
        forecast_result = forecast_service.get_forecast(df)
    except Exception:  # noqa: BLE001
        forecast_result = {}

    return {
        "source": settings.DATA_SOURCE_LABEL,
        "granularity": granularity,
        "summary": summary,
        "zone_energy": zone_totals,
        "equipment_split": eq_split,
        "forecast": forecast_result.get("forecast", []),
        "predicted_peak_kw": forecast_result.get("predicted_peak_kw"),
        "predicted_peak_time": forecast_result.get("predicted_peak_time"),
        "daily_series": energy_series(df),
    }


def energy_summary(df):
    total = float(df["energy_kwh"].sum())
    cost = total * settings.ENERGY_TARIFF_PER_KWH
    carbon = total * settings.CARBON_FACTOR_KG_PER_KWH
    p95 = float(df["power_kw"].quantile(0.95))
    peak_row = df.loc[df["power_kw"].idxmax()]
    return {
        "total_energy_kwh": round(total, 2),
        "estimated_cost_inr": round(cost, 2),
        "estimated_carbon_kg": round(carbon, 2),
        "mean_power_kw": round(float(df["power_kw"].mean()), 2),
        "peak_kw": round(float(peak_row["power_kw"]), 2),
        "peak_time": str(peak_row["timestamp"]),
        "p95_kw": round(p95, 2),
        "hours": int(len(df)),
    }


def energy_equipment(df):
    total = float(df["energy_kwh"].sum())
    parts = [
        ("HVAC", float(df["hvac_load"].sum())),
        ("Lighting", float(df["lighting_load"].sum())),
        ("Pumps", float(df["pump_load"].sum())),
        ("Other Loads", float(df["other_load"].sum())),
    ]
    return [{"name": n, "energy_kwh": round(v, 2),
             "share_pct": round(v / total * 100, 1) if total else 0}
            for n, v in parts]


def energy_series(df):
    df = df.copy()
    df["date"] = df["timestamp"].dt.date
    g = df.groupby("date").agg(energy_kwh=("energy_kwh", "sum"),
                               peak_kw=("power_kw", "max")).reset_index()
    return [{"date": str(r["date"]), "energy_kwh": round(float(r["energy_kwh"]), 2),
             "peak_kw": round(float(r["peak_kw"]), 2)} for r in g.to_dict(orient="records")]