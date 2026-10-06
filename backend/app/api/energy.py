from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.config import settings
from app.database.connection import get_db
from app.database.models import Building, Zone
from app.services import analytics_service, energy_service, data_provider

router = APIRouter()


@router.get("/buildings")
def get_buildings(db: Session = Depends(get_db)):
    buildings = db.query(Building).all()
    return {
        "source": settings.DATA_SOURCE_LABEL,
        "buildings": [
            {
                "id": b.id, "name": b.name, "location": b.location, "type": b.type,
            }
            for b in buildings
        ],
    }


@router.get("/zones")
def get_zones(db: Session = Depends(get_db), building_id: int | None = None):
    q = db.query(Zone)
    if building_id:
        q = q.filter(Zone.building_id == building_id)
    zones = q.order_by(Zone.id).all()
    return {
        "source": settings.DATA_SOURCE_LABEL,
        "zones": [
            {"id": z.id, "building_id": z.building_id, "name": z.name,
             "zone_type": z.zone_type, "area": z.area}
            for z in zones
        ],
    }


@router.get("/energy/current")
def current_energy(db: Session = Depends(get_db)):
    """Latest aggregated building state + recent trends."""
    now = datetime.now()
    window_start = now - timedelta(days=0)
    df_b = analytics_service.building_level_frame(db, start=now - timedelta(days=1))
    if df_b.empty:
        raise HTTPException(status_code=503, detail="No energy data available (database unavailable or empty).")

    last = df_b.iloc[-1]
    # today's rolling total from raw readings
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    df_today = analytics_service.readings_frame(db, start=today_start)
    today_energy = float(df_today["energy_kwh"].sum()) if not df_today.empty else 0.0

    total_yesterday = float(
        analytics_service.building_level_frame(db, start=now - timedelta(days=1),
                                               end=today_start)["energy_kwh"].sum()
    ) if not analytics_service.building_level_frame(db, start=now - timedelta(days=1), end=today_start).empty else 0.0

    df_24 = analytics_service.building_level_frame(db, start=now - timedelta(days=1))
    peak = energy_service.peak_demand(df_24)
    cc = energy_service.cost_and_carbon(today_energy)
    zones = analytics_service.zone_totals(db, start=now - timedelta(days=1))

    potential = 0.0
    if today_energy > 0:
        potential = round(today_energy * 0.08 * settings.ENERGY_TARIFF_PER_KWH, 2)

    return {
        "source": settings.DATA_SOURCE_LABEL,
        "timestamp": last["timestamp"].isoformat(),
        "current_power_kw": round(float(last["power_kw"]), 2),
        "current_hvac_kw": round(float(last["hvac_load"]), 2),
        "current_lighting_kw": round(float(last["lighting_load"]), 2),
        "current_temperature_c": round(float(last["temperature"]), 1),
        "current_occupancy_pct": round(float(last["occupancy"]), 1),
        "today_energy_kwh": round(today_energy, 2),
        "yesterday_energy_kwh": round(total_yesterday, 2),
        "peak_demand_kw": peak["peak_kw"],
        "peak_time": peak["peak_time"],
        "today_cost_inr": cc["cost_inr"],
        "today_carbon_kg": cc["carbon_kg"],
        "estimated_savings_potential_inr": potential,
        "savings_label": "ESTIMATED",
        "zone_totals_24h": zones,
    }


@router.get("/energy/history")
def energy_history(
    granularity: str = Query("hourly", pattern="^(hourly|daily|weekly|monthly)$"),
    start: datetime | None = None,
    end: datetime | None = None,
    zone_id: int | None = None,
    building_id: int | None = None,
    db: Session = Depends(get_db),
):
    df = None
    if zone_id:
        df = analytics_service.readings_frame(db, start=start, end=end, zone_id=zone_id,
                                              building_id=building_id)
    else:
        df = analytics_service.building_level_frame(db, start=start, end=end, building_id=building_id)
    if df.empty:
        return {"source": settings.DATA_SOURCE_LABEL, "data": [], "granularity": granularity}

    if granularity == "hourly":
        # cap to last 7 days for perf; full hourly otherwise
        data = df.tail(7 * 24).to_dict(orient="records")
        data = [{**r, "timestamp": str(r["timestamp"]), "source": settings.DATA_SOURCE_LABEL}
                for r in data]
    else:
        data = energy_service.resample_series(df, granularity)
    return {"source": settings.DATA_SOURCE_LABEL, "granularity": granularity, "data": data}


@router.get("/energy/by-zone")
def energy_by_zone(start: datetime | None = None, end: datetime | None = None,
                   db: Session = Depends(get_db)):
    return {
        "source": settings.DATA_SOURCE_LABEL,
        "zones": analytics_service.zone_totals(db, start=start, end=end),
    }